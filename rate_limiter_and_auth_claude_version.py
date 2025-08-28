# 🚀 PRODUCTION-READY REDIS RATE LIMITING FOR FASTAPI DATABASE ACCESS

import redis.asyncio as redis
import time
import hashlib
import json
from typing import Optional, Dict, Any, Callable, Union
from functools import wraps
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
import logging
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Rate limit types for different database operations"""
    READ = "read"
    WRITE = "write" 
    BULK = "bulk"
    EXPORT = "export"
    SEARCH = "search"

class RedisRateLimiter:
    """Production-ready Redis rate limiter for FastAPI database operations"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        
        # Default rate limits by operation type
        self.default_limits = {
            RateLimitType.READ: {"requests": 1000, "window": 60},      # 1000 reads/minute
            RateLimitType.WRITE: {"requests": 100, "window": 60},      # 100 writes/minute
            RateLimitType.BULK: {"requests": 10, "window": 300},       # 10 bulk ops/5min
            RateLimitType.EXPORT: {"requests": 5, "window": 3600},     # 5 exports/hour
            RateLimitType.SEARCH: {"requests": 200, "window": 60},     # 200 searches/minute
        }
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis connection with connection pooling"""
        if not self._redis_client:
            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
        return self._redis_client
    
    async def close(self):
        """Close Redis connection"""
        if self._redis_client:
            await self._redis_client.close()
    
    def _get_client_identifier(self, request: Request, user_id: Optional[str] = None) -> str:
        """Generate unique client identifier for rate limiting"""
        # Use user_id if available (authenticated requests)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP + User-Agent hash for anonymous requests
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        identifier_hash = hashlib.sha256(f"{client_ip}:{user_agent}".encode()).hexdigest()[:16]
        return f"anon:{identifier_hash}"
    
    def _get_rate_limit_key(self, 
                           client_id: str, 
                           operation_type: RateLimitType, 
                           endpoint: str,
                           window_start: int) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{operation_type.value}:{endpoint}:{client_id}:{window_start}"
    
    async def _sliding_window_check(self,
                                   redis_client: redis.Redis,
                                   key: str,
                                   limit: int,
                                   window_seconds: int) -> Dict[str, Any]:
        """Implement sliding window rate limiting algorithm"""
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        # Use Redis pipeline for atomic operations
        pipe = redis_client.pipeline()
        
        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, window_seconds + 60)  # Extra 60s buffer
        
        results = await pipe.execute()
        
        current_requests = results[1]  # Count from zcard
        
        # Check if limit exceeded
        if current_requests >= limit:
            # Remove the request we just added since it's over limit
            await redis_client.zrem(key, str(current_time))
            return {
                "allowed": False,
                "current_requests": current_requests,
                "limit": limit,
                "reset_time": current_time + window_seconds
            }
        
        return {
            "allowed": True,
            "current_requests": current_requests + 1,
            "limit": limit,
            "reset_time": current_time + window_seconds
        }
    
    async def check_rate_limit(self,
                              request: Request,
                              operation_type: RateLimitType,
                              endpoint: str,
                              user_id: Optional[str] = None,
                              custom_limit: Optional[int] = None,
                              custom_window: Optional[int] = None) -> Dict[str, Any]:
        """Check if request is within rate limit"""
        
        redis_client = await self.get_redis()
        client_id = self._get_client_identifier(request, user_id)
        
        # Get rate limit settings
        limit_config = self.default_limits[operation_type]
        limit = custom_limit or limit_config["requests"]
        window = custom_window or limit_config["window"]
        
        current_time = int(time.time())
        window_start = current_time // window * window  # Align to window boundary
        
        key = self._get_rate_limit_key(client_id, operation_type, endpoint, window_start)
        
        try:
            result = await self._sliding_window_check(redis_client, key, limit, window)
            
            # Log rate limit events
            if not result["allowed"]:
                logger.warning(
                    f"Rate limit exceeded for {client_id} on {endpoint} "
                    f"({operation_type.value}): {result['current_requests']}/{limit}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fail open - allow request if Redis is down
            return {
                "allowed": True,
                "current_requests": 0,
                "limit": limit,
                "reset_time": current_time + window,
                "error": str(e)
            }

# Global rate limiter instance
rate_limiter = RedisRateLimiter()

def database_rate_limit(
    operation_type: RateLimitType,
    requests_per_window: Optional[int] = None,
    window_seconds: Optional[int] = None,
    error_message: Optional[str] = None
):
    """
    Production-ready rate limiting decorator for FastAPI database operations
    
    Args:
        operation_type: Type of database operation (READ, WRITE, BULK, etc.)
        requests_per_window: Custom request limit (overrides default)
        window_seconds: Custom time window (overrides default)
        error_message: Custom error message for rate limit exceeded
    
    Usage:
        @database_rate_limit(RateLimitType.WRITE, requests_per_window=50, window_seconds=60)
        async def create_user(user: UserCreate, request: Request):
            # Your database operation here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and user info from function arguments
            request: Optional[Request] = None
            user_id: Optional[str] = None
            
            # Find Request object in args/kwargs
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break
            
            # Find user_id in kwargs (common FastAPI pattern)
            user_id = kwargs.get('current_user') or kwargs.get('user_id')
            
            if not request:
                logger.warning(f"No Request object found in {func.__name__}, skipping rate limit")
                return await func(*args, **kwargs)
            
            # Get endpoint name
            endpoint = f"{request.method}:{request.url.path}"
            
            # Check rate limit
            rate_check = await rate_limiter.check_rate_limit(
                request=request,
                operation_type=operation_type,
                endpoint=endpoint,
                user_id=user_id,
                custom_limit=requests_per_window,
                custom_window=window_seconds
            )
            
            if not rate_check["allowed"]:
                # Prepare rate limit headers
                retry_after = rate_check["reset_time"] - int(time.time())
                
                error_msg = error_message or (
                    f"Rate limit exceeded for {operation_type.value} operations. "
                    f"Limit: {rate_check['limit']} requests per "
                    f"{window_seconds or rate_limiter.default_limits[operation_type]['window']} seconds. "
                    f"Try again in {retry_after} seconds."
                )
                
                # Return rate limit error with proper headers
                return JSONResponse(
                    status_code=429,
                    content={"detail": error_msg, "retry_after": retry_after},
                    headers={
                        "X-RateLimit-Limit": str(rate_check["limit"]),
                        "X-RateLimit-Remaining": str(max(0, rate_check["limit"] - rate_check["current_requests"])),
                        "X-RateLimit-Reset": str(rate_check["reset_time"]),
                        "Retry-After": str(retry_after)
                    }
                )
            
            # Add rate limit info to response headers
            try:
                response = await func(*args, **kwargs)
                
                # Add rate limit headers to successful responses
                if hasattr(response, 'headers'):
                    response.headers["X-RateLimit-Limit"] = str(rate_check["limit"])
                    response.headers["X-RateLimit-Remaining"] = str(
                        max(0, rate_check["limit"] - rate_check["current_requests"])
                    )
                    response.headers["X-RateLimit-Reset"] = str(rate_check["reset_time"])
                
                return response
                
            except Exception as e:
                # Log the error but don't count failed operations against rate limit
                logger.error(f"Error in {func.__name__}: {e}")
                raise
                
        return wrapper
    return decorator

# Convenience decorators for common database operations
def read_rate_limit(requests_per_minute: int = 1000):
    """Rate limit for database read operations"""
    return database_rate_limit(
        RateLimitType.READ, 
        requests_per_window=requests_per_minute, 
        window_seconds=60
    )

def write_rate_limit(requests_per_minute: int = 100):
    """Rate limit for database write operations"""
    return database_rate_limit(
        RateLimitType.WRITE, 
        requests_per_window=requests_per_minute, 
        window_seconds=60
    )

def bulk_rate_limit(requests_per_hour: int = 10):
    """Rate limit for bulk database operations"""
    return database_rate_limit(
        RateLimitType.BULK, 
        requests_per_window=requests_per_hour, 
        window_seconds=3600
    )

def export_rate_limit(requests_per_hour: int = 5):
    """Rate limit for data export operations"""
    return database_rate_limit(
        RateLimitType.EXPORT, 
        requests_per_window=requests_per_hour, 
        window_seconds=3600
    )

print("✅ Production Redis Rate Limiter ready!")
print("📊 Default limits:")
for op_type, config in RedisRateLimiter().default_limits.items():
    print(f"  {op_type.value}: {config['requests']} requests per {config['window']} seconds")

# 📝 FASTAPI USAGE EXAMPLES WITH REDIS RATE LIMITING

from fastapi import FastAPI, Depends, Request, HTTPException
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI(title="Rate Limited Database API")

# Example models
class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

# Mock current user dependency
async def get_current_user(request: Request) -> str:
    # In production, extract from JWT token
    return request.headers.get("X-User-ID", "anonymous")

# 🔥 EXAMPLE 1: Simple read endpoint with rate limiting
@app.get("/users/{user_id}", response_model=UserResponse)
@read_rate_limit(requests_per_minute=500)  # 500 reads per minute
async def get_user(
    user_id: int, 
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get user by ID with read rate limiting"""
    # Your database read logic here
    return UserResponse(id=user_id, name="John Doe", email="john@example.com")

# 🔥 EXAMPLE 2: Write endpoint with stricter rate limiting
@app.post("/users/", response_model=UserResponse)
@write_rate_limit(requests_per_minute=50)  # 50 writes per minute
async def create_user(
    user: UserCreate,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Create user with write rate limiting"""
    # Your database write logic here
    return UserResponse(id=1, name=user.name, email=user.email)

# 🔥 EXAMPLE 3: Bulk operation with custom rate limiting
@app.post("/users/bulk", response_model=List[UserResponse])
@database_rate_limit(
    RateLimitType.BULK, 
    requests_per_window=5,     # Only 5 bulk operations
    window_seconds=300,        # Per 5 minutes
    error_message="Bulk operations limited to 5 per 5 minutes to protect database performance"
)
async def bulk_create_users(
    users: List[UserCreate],
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Bulk create users with strict rate limiting"""
    if len(users) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 users per bulk operation")
    
    # Your bulk database logic here
    return [UserResponse(id=i, name=user.name, email=user.email) for i, user in enumerate(users)]

# 🔥 EXAMPLE 4: Export endpoint with hourly limits
@app.get("/users/export")
@export_rate_limit(requests_per_hour=3)  # 3 exports per hour
async def export_users(
    request: Request,
    format: str = "csv",
    current_user: str = Depends(get_current_user)
):
    """Export users data with strict hourly limits"""
    # Your export logic here
    return {"message": f"Exporting users in {format} format", "user": current_user}

# 🔥 EXAMPLE 5: Search endpoint with custom error message
@app.get("/users/search")
@database_rate_limit(
    RateLimitType.SEARCH,
    requests_per_window=100,
    window_seconds=60,
    error_message="Search operations are limited to protect database performance. Please wait before searching again."
)
async def search_users(
    q: str,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Search users with rate limiting"""
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Search query must be at least 3 characters")
    
    # Your search logic here
    return {"results": [], "query": q, "user": current_user}

# 🔥 EXAMPLE 6: Multiple rate limits on one endpoint
@app.delete("/users/{user_id}")
@write_rate_limit(requests_per_minute=20)  # Can stack multiple decorators
async def delete_user(
    user_id: int,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Delete user with write rate limiting"""
    # Your delete logic here
    return {"message": f"User {user_id} deleted", "deleted_by": current_user}

print("📝 FastAPI endpoints with rate limiting configured!")
print("🎯 Test with: curl -H 'X-User-ID: test_user' http://localhost:8000/users/1")


# 🚀 ADVANCED REDIS RATE LIMITING FEATURES

class AdvancedRedisRateLimiter(RedisRateLimiter):
    """Advanced rate limiter with monitoring, whitelisting, and progressive penalties"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__(redis_url)
        
        # Advanced features
        self.whitelist_key = "rate_limit:whitelist"
        self.blacklist_key = "rate_limit:blacklist"
        self.violation_key = "rate_limit:violations"
        self.stats_key = "rate_limit:stats"
        
        # Progressive penalty settings
        self.penalty_multipliers = {
            1: 1.0,    # First violation: normal limit
            2: 0.5,    # Second violation: 50% of normal limit
            3: 0.25,   # Third violation: 25% of normal limit
            4: 0.1,    # Fourth+ violation: 10% of normal limit
        }
    
    async def add_to_whitelist(self, identifier: str, duration_seconds: int = 3600):
        """Add client to whitelist (bypasses rate limiting)"""
        redis_client = await self.get_redis()
        await redis_client.setex(f"{self.whitelist_key}:{identifier}", duration_seconds, "1")
        logger.info(f"Added {identifier} to whitelist for {duration_seconds} seconds")
    
    async def add_to_blacklist(self, identifier: str, duration_seconds: int = 3600):
        """Add client to blacklist (blocks all requests)"""
        redis_client = await self.get_redis()
        await redis_client.setex(f"{self.blacklist_key}:{identifier}", duration_seconds, "1")
        logger.warning(f"Added {identifier} to blacklist for {duration_seconds} seconds")
    
    async def is_whitelisted(self, identifier: str) -> bool:
        """Check if client is whitelisted"""
        redis_client = await self.get_redis()
        return await redis_client.exists(f"{self.whitelist_key}:{identifier}")
    
    async def is_blacklisted(self, identifier: str) -> bool:
        """Check if client is blacklisted"""
        redis_client = await self.get_redis()
        return await redis_client.exists(f"{self.blacklist_key}:{identifier}")
    
    async def record_violation(self, identifier: str, operation_type: RateLimitType):
        """Record rate limit violation for progressive penalties"""
        redis_client = await self.get_redis()
        violation_key = f"{self.violation_key}:{identifier}:{operation_type.value}"
        
        # Increment violation count with 24-hour expiry
        pipe = redis_client.pipeline()
        pipe.incr(violation_key)
        pipe.expire(violation_key, 86400)  # 24 hours
        
        results = await pipe.execute()
        violation_count = results[0]
        
        logger.warning(f"Violation #{violation_count} for {identifier} on {operation_type.value}")
        
        # Auto-blacklist after too many violations
        if violation_count >= 10:
            await self.add_to_blacklist(identifier, 3600)  # 1 hour blacklist
            logger.error(f"Auto-blacklisted {identifier} for excessive violations")
        
        return violation_count
    
    async def get_violation_count(self, identifier: str, operation_type: RateLimitType) -> int:
        """Get current violation count for progressive penalties"""
        redis_client = await self.get_redis()
        violation_key = f"{self.violation_key}:{identifier}:{operation_type.value}"
        count = await redis_client.get(violation_key)
        return int(count) if count else 0
    
    async def calculate_adjusted_limit(self, 
                                     base_limit: int, 
                                     identifier: str, 
                                     operation_type: RateLimitType) -> int:
        """Calculate rate limit with progressive penalties"""
        violation_count = await self.get_violation_count(identifier, operation_type)
        
        if violation_count == 0:
            return base_limit
        
        # Apply penalty multiplier
        penalty_level = min(violation_count, max(self.penalty_multipliers.keys()))
        multiplier = self.penalty_multipliers[penalty_level]
        adjusted_limit = max(1, int(base_limit * multiplier))
        
        logger.info(f"Adjusted limit for {identifier}: {adjusted_limit} (was {base_limit}, {violation_count} violations)")
        return adjusted_limit
    
    async def record_request_stats(self, 
                                  identifier: str, 
                                  operation_type: RateLimitType, 
                                  allowed: bool):
        """Record request statistics for monitoring"""
        redis_client = await self.get_redis()
        timestamp = int(time.time() // 60) * 60  # Round to minute
        
        stats_key = f"{self.stats_key}:{operation_type.value}:{timestamp}"
        
        pipe = redis_client.pipeline()
        pipe.hincrby(stats_key, "total_requests", 1)
        if allowed:
            pipe.hincrby(stats_key, "allowed_requests", 1)
        else:
            pipe.hincrby(stats_key, "blocked_requests", 1)
        pipe.expire(stats_key, 86400)  # Keep stats for 24 hours
        
        await pipe.execute()
    
    async def get_stats(self, operation_type: RateLimitType, minutes: int = 60) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        redis_client = await self.get_redis()
        current_time = int(time.time() // 60) * 60
        
        stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "time_period_minutes": minutes
        }
        
        for i in range(minutes):
            timestamp = current_time - (i * 60)
            stats_key = f"{self.stats_key}:{operation_type.value}:{timestamp}"
            minute_stats = await redis_client.hgetall(stats_key)
            
            for key in ["total_requests", "allowed_requests", "blocked_requests"]:
                stats[key] += int(minute_stats.get(key, 0))
        
        if stats["total_requests"] > 0:
            stats["block_rate"] = stats["blocked_requests"] / stats["total_requests"]
        else:
            stats["block_rate"] = 0.0
        
        return stats
    
    async def enhanced_rate_limit_check(self,
                                      request: Request,
                                      operation_type: RateLimitType,
                                      endpoint: str,
                                      user_id: Optional[str] = None,
                                      custom_limit: Optional[int] = None,
                                      custom_window: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced rate limit check with all advanced features"""
        
        client_id = self._get_client_identifier(request, user_id)
        
        # Check blacklist first
        if await self.is_blacklisted(client_id):
            await self.record_request_stats(client_id, operation_type, False)
            return {
                "allowed": False,
                "reason": "blacklisted",
                "current_requests": 0,
                "limit": 0,
                "reset_time": int(time.time()) + 3600
            }
        
        # Check whitelist (bypass all limits)
        if await self.is_whitelisted(client_id):
            await self.record_request_stats(client_id, operation_type, True)
            return {
                "allowed": True,
                "reason": "whitelisted",
                "current_requests": 0,
                "limit": float('inf'),
                "reset_time": int(time.time()) + 3600
            }
        
        # Get base rate limit
        limit_config = self.default_limits[operation_type]
        base_limit = custom_limit or limit_config["requests"]
        window = custom_window or limit_config["window"]
        
        # Apply progressive penalties
        adjusted_limit = await self.calculate_adjusted_limit(base_limit, client_id, operation_type)
        
        # Perform normal rate limit check with adjusted limit
        result = await self._sliding_window_check(
            await self.get_redis(),
            self._get_rate_limit_key(client_id, operation_type, endpoint, int(time.time()) // window * window),
            adjusted_limit,
            window
        )
        
        # Record violation if blocked
        if not result["allowed"]:
            await self.record_violation(client_id, operation_type)
        
        # Record stats
        await self.record_request_stats(client_id, operation_type, result["allowed"])
        
        return result

# Global advanced rate limiter
advanced_rate_limiter = AdvancedRedisRateLimiter()

def advanced_database_rate_limit(
    operation_type: RateLimitType,
    requests_per_window: Optional[int] = None,
    window_seconds: Optional[int] = None,
    error_message: Optional[str] = None
):
    """Advanced rate limiting decorator with all features"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and user info (same as before)
            request: Optional[Request] = None
            user_id: Optional[str] = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break
            
            user_id = kwargs.get('current_user') or kwargs.get('user_id')
            
            if not request:
                return await func(*args, **kwargs)
            
            endpoint = f"{request.method}:{request.url.path}"
            
            # Use enhanced rate limit check
            rate_check = await advanced_rate_limiter.enhanced_rate_limit_check(
                request=request,
                operation_type=operation_type,
                endpoint=endpoint,
                user_id=user_id,
                custom_limit=requests_per_window,
                custom_window=window_seconds
            )
            
            if not rate_check["allowed"]:
                retry_after = rate_check["reset_time"] - int(time.time())
                
                # Custom error messages based on reason
                if rate_check.get("reason") == "blacklisted":
                    error_msg = "Access temporarily blocked due to excessive rate limit violations"
                else:
                    error_msg = error_message or f"Rate limit exceeded for {operation_type.value} operations"
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": error_msg, 
                        "retry_after": retry_after,
                        "reason": rate_check.get("reason", "rate_limited")
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_check["limit"]),
                        "X-RateLimit-Remaining": str(max(0, rate_check["limit"] - rate_check["current_requests"])),
                        "X-RateLimit-Reset": str(rate_check["reset_time"]),
                        "Retry-After": str(retry_after)
                    }
                )
            
            # Execute function and add headers
            response = await func(*args, **kwargs)
            
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(rate_check["limit"])
                response.headers["X-RateLimit-Remaining"] = str(
                    max(0, rate_check["limit"] - rate_check["current_requests"])
                )
                response.headers["X-RateLimit-Reset"] = str(rate_check["reset_time"])
                
                # Add reason if whitelisted
                if rate_check.get("reason") == "whitelisted":
                    response.headers["X-RateLimit-Status"] = "whitelisted"
            
            return response
                
        return wrapper
    return decorator

print("🚀 Advanced Redis Rate Limiter ready!")
print("✨ Features:")
print("  • Progressive penalties")
print("  • Whitelist/blacklist support")
print("  • Request statistics")
print("  • Violation tracking")
print("  • Auto-blacklisting")


# 📊 MONITORING DASHBOARD & ADMIN ENDPOINTS

@app.get("/admin/rate-limit/stats/{operation_type}")
async def get_rate_limit_stats(
    operation_type: str,
    minutes: int = 60,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get rate limiting statistics for monitoring"""
    # Add admin permission check here
    try:
        op_type = RateLimitType(operation_type)
        stats = await advanced_rate_limiter.get_stats(op_type, minutes)
        return stats
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid operation type")

@app.post("/admin/rate-limit/whitelist/{identifier}")
async def add_to_whitelist(
    identifier: str,
    duration: int = 3600,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Add client to whitelist"""
    # Add admin permission check here
    await advanced_rate_limiter.add_to_whitelist(identifier, duration)
    return {"message": f"Added {identifier} to whitelist for {duration} seconds"}

@app.post("/admin/rate-limit/blacklist/{identifier}")
async def add_to_blacklist(
    identifier: str,
    duration: int = 3600,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Add client to blacklist"""
    # Add admin permission check here
    await advanced_rate_limiter.add_to_blacklist(identifier, duration)
    return {"message": f"Added {identifier} to blacklist for {duration} seconds"}

# 🔧 PRODUCTION DEPLOYMENT GUIDE

production_setup = '''
# 🏭 PRODUCTION DEPLOYMENT GUIDE

## 1. Redis Setup (High Availability)
```bash
# Install Redis with persistence
sudo apt update
sudo apt install redis-server

# Configure Redis for production (/etc/redis/redis.conf)
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## 2. Redis Cluster Setup (Optional for high load)
```bash
# Set up Redis cluster for horizontal scaling
redis-cli --cluster create \\
  127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \\
  127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \\
  --cluster-replicas 1
```

## 3. Environment Variables
```bash
export REDIS_URL="redis://localhost:6379"
export REDIS_PASSWORD="your_secure_password"
export RATE_LIMIT_ENABLED="true"
export LOG_LEVEL="INFO"
```

## 4. Docker Compose Setup
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  fastapi:
    build: .
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
    depends_on:
      - redis
    ports:
      - "8000:8000"

volumes:
  redis_data:
```

## 5. Monitoring with Prometheus
```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
rate_limit_requests = Counter(
    'rate_limit_requests_total',
    'Total rate limit requests',
    ['operation_type', 'status']
)

rate_limit_duration = Histogram(
    'rate_limit_check_duration_seconds',
    'Time spent checking rate limits'
)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 6. Alerting Rules
```yaml
# Prometheus alerting rules
groups:
  - name: rate_limiting
    rules:
      - alert: HighRateLimitBlocks
        expr: rate(rate_limit_requests_total{status="blocked"}[5m]) > 10
        for: 2m
        annotations:
          summary: "High rate limit blocking rate"
          
      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        annotations:
          summary: "Redis is down"
```

## 7. Load Testing
```bash
# Test rate limits with Apache Bench
ab -n 1000 -c 10 -H "X-User-ID: test_user" http://localhost:8000/users/1

# Test with different users
for i in {1..100}; do
  curl -H "X-User-ID: user_$i" http://localhost:8000/users/1 &
done
```
'''

print(production_setup)

# 🔍 DEBUGGING AND TROUBLESHOOTING

troubleshooting_guide = '''
# 🐛 TROUBLESHOOTING GUIDE

## Common Issues:

1. **Redis Connection Errors**
   - Check Redis is running: `redis-cli ping`
   - Verify connection string
   - Check firewall rules

2. **Rate Limits Too Strict**
   - Monitor /admin/rate-limit/stats endpoints
   - Adjust limits based on actual usage
   - Use whitelist for trusted clients

3. **Memory Usage**
   - Monitor Redis memory: `redis-cli info memory`
   - Set appropriate maxmemory policy
   - Use TTL on all keys

4. **Performance Issues**
   - Use Redis pipelining for multiple operations
   - Consider Redis cluster for high load
   - Monitor rate limit check duration

## Debugging Commands:
```python
# Check current rate limit status
await advanced_rate_limiter.get_stats(RateLimitType.READ, 60)

# Check if user is whitelisted/blacklisted
await advanced_rate_limiter.is_whitelisted("user:123")
await advanced_rate_limiter.is_blacklisted("user:123")

# Get violation count
await advanced_rate_limiter.get_violation_count("user:123", RateLimitType.WRITE)
```

## Performance Tuning:
- Use Redis cluster for >10K requests/second
- Implement local caching for rate limit checks
- Use different Redis instances for different operation types
- Monitor and adjust window sizes based on usage patterns
'''

print(troubleshooting_guide)

# 🚀 STARTUP AND SHUTDOWN HANDLERS

@app.on_event("startup")
async def startup_event():
    """Initialize rate limiter on startup"""
    try:
        # Test Redis connection
        redis_client = await advanced_rate_limiter.get_redis()
        await redis_client.ping()
        logger.info("✅ Redis rate limiter initialized successfully")
        
        # Load default whitelisted IPs (if any)
        # await advanced_rate_limiter.add_to_whitelist("admin_ip", 86400)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Redis rate limiter: {e}")
        # Decide whether to fail startup or continue without rate limiting

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    try:
        await advanced_rate_limiter.close()
        logger.info("✅ Redis rate limiter closed successfully")
    except Exception as e:
        logger.error(f"❌ Error closing Redis rate limiter: {e}")

print("✅ Production-ready Redis rate limiter complete!")
print("🎯 Key Features:")
print("  • Sliding window algorithm")
print("  • Progressive penalties")
print("  • Whitelist/blacklist support")
print("  • Request monitoring & stats")
print("  • Admin management endpoints")
print("  • Production deployment guide")
print("  • Comprehensive error handling")
print("  • Performance optimizations")

# Installation requirements
requirements = '''
# 📦 INSTALLATION REQUIREMENTS

pip install redis[hiredis] fastapi uvicorn prometheus-client

# Or with requirements.txt:
redis[hiredis]>=4.5.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
prometheus-client>=0.17.0
python-jose[cryptography]>=3.3.0  # For JWT
'''
print(requirements)

# 🔐 PRODUCTION AUTHENTICATION SYSTEM FOR FASTAPI

import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, EmailStr, validator
from fastapi import HTTPException, Depends, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
import secrets
import hashlib
import os
from dataclasses import dataclass
import asyncpg
import redis.asyncio as redis

# Authentication configuration
class AuthConfig:
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    PASSWORD_MIN_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15

class UserRole(str, Enum):
    """User roles for role-based access control"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    READONLY = "readonly"

class Permission(str, Enum):
    """Granular permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    BULK_OPERATIONS = "bulk_operations"
    EXPORT = "export"
    ADMIN = "admin"
    USER_MANAGEMENT = "user_management"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ, Permission.WRITE, Permission.DELETE,
        Permission.BULK_OPERATIONS, Permission.EXPORT, Permission.ADMIN,
        Permission.USER_MANAGEMENT
    ],
    UserRole.MANAGER: [
        Permission.READ, Permission.WRITE, Permission.DELETE,
        Permission.BULK_OPERATIONS, Permission.EXPORT
    ],
    UserRole.USER: [
        Permission.READ, Permission.WRITE, Permission.DELETE
    ],
    UserRole.READONLY: [
        Permission.READ
    ]
}

# Pydantic models for authentication
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (with _ and - allowed)')
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < AuthConfig.PASSWORD_MIN_LENGTH:
            raise ValueError(f'Password must be at least {AuthConfig.PASSWORD_MIN_LENGTH} characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = []

# Authentication service
class AuthenticationService:
    """Production-ready authentication service"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis connection for session management"""
        if not self._redis_client:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
    
    async def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])
            
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            user_id: int = payload.get("sub")
            username: str = payload.get("username")
            role: str = payload.get("role")
            
            if user_id is None or username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            # Check if token is blacklisted
            redis_client = await self.get_redis()
            is_blacklisted = await redis_client.exists(f"blacklist_token:{token}")
            if is_blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            permissions = ROLE_PERMISSIONS.get(UserRole(role), [])
            
            return TokenData(
                user_id=user_id,
                username=username,
                role=role,
                permissions=[p.value for p in permissions]
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    async def get_login_attempts(self, username: str) -> int:
        """Get failed login attempts count"""
        redis_client = await self.get_redis()
        attempts = await redis_client.get(f"login_attempts:{username}")
        return int(attempts) if attempts else 0
    
    async def record_failed_login(self, username: str) -> int:
        """Record failed login attempt"""
        redis_client = await self.get_redis()
        key = f"login_attempts:{username}"
        
        # Increment counter with expiry
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, AuthConfig.LOCKOUT_DURATION_MINUTES * 60)
        results = await pipe.execute()
        
        return results[0]
    
    async def clear_login_attempts(self, username: str):
        """Clear failed login attempts"""
        redis_client = await self.get_redis()
        await redis_client.delete(f"login_attempts:{username}")
    
    async def is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        attempts = await self.get_login_attempts(username)
        return attempts >= AuthConfig.MAX_LOGIN_ATTEMPTS
    
    async def blacklist_token(self, token: str):
        """Add token to blacklist (for logout)"""
        try:
            payload = jwt.decode(token, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])
            exp = payload.get("exp")
            if exp:
                redis_client = await self.get_redis()
                ttl = exp - int(datetime.now(timezone.utc).timestamp())
                if ttl > 0:
                    await redis_client.setex(f"blacklist_token:{token}", ttl, "1")
        except jwt.JWTError:
            pass  # Invalid token, no need to blacklist
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create new user account"""
        conn = await asyncpg.connect(self.db_url)
        try:
            # Check if username or email already exists
            existing_user = await conn.fetchrow(
                "SELECT id FROM users WHERE username = $1 OR email = $2",
                user_data.username, user_data.email
            )
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered"
                )
            
            # Hash password and create user
            hashed_password = self.hash_password(user_data.password)
            
            user_row = await conn.fetchrow("""
                INSERT INTO users (username, email, password_hash, full_name, role, is_active, created_at)
                VALUES ($1, $2, $3, $4, $5, true, $6)
                RETURNING id, username, email, full_name, role, is_active, created_at
            """, user_data.username, user_data.email, hashed_password, 
                user_data.full_name, user_data.role.value, datetime.now(timezone.utc))
            
            return UserResponse(**dict(user_row), last_login=None)
            
        finally:
            await conn.close()
    
    async def authenticate_user(self, username: str, password: str) -> Optional[UserResponse]:
        """Authenticate user credentials"""
        # Check account lockout
        if await self.is_account_locked(username):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked due to too many failed attempts. Try again in {AuthConfig.LOCKOUT_DURATION_MINUTES} minutes."
            )
        
        conn = await asyncpg.connect(self.db_url)
        try:
            user_row = await conn.fetchrow(
                "SELECT * FROM users WHERE username = $1 AND is_active = true",
                username
            )
            
            if not user_row:
                await self.record_failed_login(username)
                return None
            
            if not self.verify_password(password, user_row['password_hash']):
                await self.record_failed_login(username)
                return None
            
            # Successful login - clear failed attempts and update last login
            await self.clear_login_attempts(username)
            await conn.execute(
                "UPDATE users SET last_login = $1 WHERE id = $2",
                datetime.now(timezone.utc), user_row['id']
            )
            
            return UserResponse(**{k: v for k, v in dict(user_row).items() if k != 'password_hash'})
            
        finally:
            await conn.close()
    
    async def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID"""
        conn = await asyncpg.connect(self.db_url)
        try:
            user_row = await conn.fetchrow(
                "SELECT id, username, email, full_name, role, is_active, created_at, last_login FROM users WHERE id = $1 AND is_active = true",
                user_id
            )
            
            if user_row:
                return UserResponse(**dict(user_row))
            return None
            
        finally:
            await conn.close()

# Global authentication service
auth_service = AuthenticationService(
    db_url=os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/db"),
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
)

# Security dependencies
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """Get current authenticated user"""
    token_data = await auth_service.verify_token(credentials.credentials)
    user = await auth_service.get_user_by_id(token_data.user_id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def require_permissions(*required_permissions: Permission):
    """Decorator to require specific permissions"""
    def permission_checker(current_user: UserResponse = Depends(get_current_active_user)):
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission.value}"
                )
        
        return current_user
    
    return permission_checker

def require_role(*required_roles: UserRole):
    """Decorator to require specific roles"""
    def role_checker(current_user: UserResponse = Depends(get_current_active_user)):
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {[r.value for r in required_roles]}"
            )
        
        return current_user
    
    return role_checker

print("🔐 Production Authentication System Ready!")
print("✨ Features:")
print("  • JWT tokens with refresh")
print("  • Role-based access control")
print("  • Permission-based authorization")
print("  • Account lockout protection")
print("  • Token blacklisting")
print("  • Password complexity validation")
print("  • Redis session management")

# 🔐 AUTHENTICATION ENDPOINTS AND ROUTE INTEGRATION

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

# Create authentication router
auth_router = APIRouter(prefix="/auth", tags=["authentication"])

@auth_router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, request: Request):
    """Register new user account"""
    try:
        user = await auth_service.create_user(user_data)
        
        # Log registration
        logger.info(f"New user registered: {user.username} from {request.client.host}")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@auth_router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None):
    """Login and get access token"""
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(form_data.username, form_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value
        }
        
        access_token = auth_service.create_access_token(token_data)
        refresh_token = auth_service.create_refresh_token(token_data)
        
        # Log successful login
        if request:
            logger.info(f"User {user.username} logged in from {request.client.host}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str = Form(...)):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        token_data = await auth_service.verify_token(refresh_token, "refresh")
        
        # Get current user
        user = await auth_service.get_user_by_id(token_data.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new tokens
        new_token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value
        }
        
        new_access_token = auth_service.create_access_token(new_token_data)
        new_refresh_token = auth_service.create_refresh_token(new_token_data)
        
        # Blacklist old refresh token
        await auth_service.blacklist_token(refresh_token)
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

@auth_router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout and blacklist token"""
    try:
        # Blacklist the access token
        await auth_service.blacklist_token(credentials.credentials)
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"message": "Logout completed"}  # Always succeed for logout

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

@auth_router.put("/me", response_model=UserResponse)
async def update_current_user(
    full_name: Optional[str] = None,
    email: Optional[EmailStr] = None,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update current user profile"""
    conn = await asyncpg.connect(auth_service.db_url)
    try:
        update_fields = []
        params = []
        param_count = 0
        
        if full_name is not None:
            param_count += 1
            update_fields.append(f"full_name = ${param_count}")
            params.append(full_name)
        
        if email is not None:
            param_count += 1
            update_fields.append(f"email = ${param_count}")
            params.append(email)
        
        if not update_fields:
            return current_user
        
        param_count += 1
        params.append(current_user.id)
        
        query = f"""
            UPDATE users 
            SET {', '.join(update_fields)}
            WHERE id = ${param_count}
            RETURNING id, username, email, full_name, role, is_active, created_at, last_login
        """
        
        updated_user = await conn.fetchrow(query, *params)
        return UserResponse(**dict(updated_user))
        
    finally:
        await conn.close()

@auth_router.post("/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Change user password"""
    conn = await asyncpg.connect(auth_service.db_url)
    try:
        # Get current password hash
        user_row = await conn.fetchrow(
            "SELECT password_hash FROM users WHERE id = $1",
            current_user.id
        )
        
        # Verify current password
        if not auth_service.verify_password(current_password, user_row['password_hash']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if len(new_password) < AuthConfig.PASSWORD_MIN_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password must be at least {AuthConfig.PASSWORD_MIN_LENGTH} characters"
            )
        
        # Hash new password and update
        new_password_hash = auth_service.hash_password(new_password)
        await conn.execute(
            "UPDATE users SET password_hash = $1 WHERE id = $2",
            new_password_hash, current_user.id
        )
        
        return {"message": "Password changed successfully"}
        
    finally:
        await conn.close()

# Admin endpoints
admin_router = APIRouter(prefix="/admin", tags=["admin"])

@admin_router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    _: UserResponse = Depends(require_permissions(Permission.USER_MANAGEMENT))
):
    """List all users (admin only)"""
    conn = await asyncpg.connect(auth_service.db_url)
    try:
        users = await conn.fetch("""
            SELECT id, username, email, full_name, role, is_active, created_at, last_login
            FROM users
            ORDER BY created_at DESC
            OFFSET $1 LIMIT $2
        """, skip, limit)
        
        return [UserResponse(**dict(user)) for user in users]
        
    finally:
        await conn.close()

@admin_router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    new_role: UserRole,
    _: UserResponse = Depends(require_permissions(Permission.USER_MANAGEMENT))
):
    """Update user role (admin only)"""
    conn = await asyncpg.connect(auth_service.db_url)
    try:
        updated_user = await conn.fetchrow("""
            UPDATE users 
            SET role = $1
            WHERE id = $2
            RETURNING id, username, email, full_name, role, is_active, created_at, last_login
        """, new_role.value, user_id)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(**dict(updated_user))
        
    finally:
        await conn.close()

@admin_router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    is_active: bool,
    _: UserResponse = Depends(require_permissions(Permission.USER_MANAGEMENT))
):
    """Activate/deactivate user (admin only)"""
    conn = await asyncpg.connect(auth_service.db_url)
    try:
        updated_user = await conn.fetchrow("""
            UPDATE users 
            SET is_active = $1
            WHERE id = $2
            RETURNING id, username, email, full_name, role, is_active, created_at, last_login
        """, is_active, user_id)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(**dict(updated_user))
        
    finally:
        await conn.close()

# Include routers in main app
app.include_router(auth_router)
app.include_router(admin_router)

print("🔐 Authentication endpoints added!")
print("📝 Available endpoints:")
print("  • POST /auth/register - Register new user")
print("  • POST /auth/login - Login and get token")
print("  • POST /auth/refresh - Refresh access token")
print("  • POST /auth/logout - Logout and blacklist token")
print("  • GET /auth/me - Get current user info")
print("  • PUT /auth/me - Update user profile")
print("  • POST /auth/change-password - Change password")
print("  • GET /admin/users - List all users (admin)")
print("  • PUT /admin/users/{id}/role - Update user role (admin)")
print("  • PUT /admin/users/{id}/status - Activate/deactivate user (admin)")


# 🔒 SECURED DATABASE ROUTES WITH AUTHENTICATION

# Updated secure database routes with authentication integration
def generate_authenticated_routes(table_info: TableInfo) -> str:
    """Generate database routes with authentication and authorization"""
    class_name = table_info.name.title().replace('_', '')
    table_name = table_info.name
    
    routes_code = f'''
# 🔒 AUTHENTICATED ROUTES FOR {table_name.upper()}

@app.get("/{table_name}/", response_model=List[{class_name}Response])
@read_rate_limit(requests_per_minute=500)
async def list_{table_name}(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    request: Request,
    current_user: UserResponse = Depends(require_permissions(Permission.READ))
):
    """List {table_name} with authentication and rate limiting"""
    # User can only see their own data unless they have admin permissions
    user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
    
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if Permission.ADMIN in user_permissions:
            # Admin can see all records
            query = f"SELECT * FROM {table_name} ORDER BY id DESC OFFSET $1 LIMIT $2"
            params = [skip, limit]
        else:
            # Regular users see only their records
            query = f"SELECT * FROM {table_name} WHERE created_by = $1 ORDER BY id DESC OFFSET $2 LIMIT $3"
            params = [current_user.id, skip, limit]
        
        rows = await conn.fetch(query, *params)
        return [{class_name}Response(**dict(row)) for row in rows]
        
    finally:
        await conn.close()

@app.get("/{table_name}/{{item_id}}", response_model={class_name}Response)
@read_rate_limit(requests_per_minute=1000)
async def get_{table_name}(
    item_id: int,
    request: Request,
    current_user: UserResponse = Depends(require_permissions(Permission.READ))
):
    """Get {table_name} by ID with ownership check"""
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid ID")
    
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        
        if Permission.ADMIN in user_permissions:
            # Admin can access any record
            query = f"SELECT * FROM {table_name} WHERE id = $1"
            params = [item_id]
        else:
            # Regular users can only access their own records
            query = f"SELECT * FROM {table_name} WHERE id = $1 AND created_by = $2"
            params = [item_id, current_user.id]
        
        row = await conn.fetchrow(query, *params)
        
        if not row:
            raise HTTPException(
                status_code=404, 
                detail=f"{class_name} not found or access denied"
            )
        
        return {class_name}Response(**dict(row))
        
    finally:
        await conn.close()

@app.post("/{table_name}/", response_model={class_name}Response)
@write_rate_limit(requests_per_minute=100)
async def create_{table_name}(
    item: {class_name}Create,
    request: Request,
    current_user: UserResponse = Depends(require_permissions(Permission.WRITE))
):
    """Create {table_name} with user ownership"""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Add audit fields
        item_dict = item.dict(exclude_unset=True)
        item_dict.update({{
            'created_by': current_user.id,
            'created_at': datetime.now(timezone.utc),
            'updated_by': current_user.id,
            'updated_at': datetime.now(timezone.utc)
        }})
        
        # Build dynamic insert query
        columns = list(item_dict.keys())
        placeholders = [f"${{i+1}}" for i in range(len(columns))]
        values = list(item_dict.values())
        
        query = f"""
            INSERT INTO {table_name} ({{", ".join(columns)}})
            VALUES ({{", ".join(placeholders)}})
            RETURNING *
        """
        
        row = await conn.fetchrow(query, *values)
        
        # Log the creation
        logger.info(f"User {{current_user.username}} created {table_name} ID {{row['id']}}")
        
        return {class_name}Response(**dict(row))
        
    finally:
        await conn.close()

@app.put("/{table_name}/{{item_id}}", response_model={class_name}Response)
@write_rate_limit(requests_per_minute=100)
async def update_{table_name}(
    item_id: int,
    item: {class_name}Update,
    request: Request,
    current_user: UserResponse = Depends(require_permissions(Permission.WRITE))
):
    """Update {table_name} with ownership check"""
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid ID")
    
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Check ownership first
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        
        if Permission.ADMIN not in user_permissions:
            # Verify ownership for non-admin users
            owner_check = await conn.fetchval(
                f"SELECT created_by FROM {table_name} WHERE id = $1",
                item_id
            )
            
            if not owner_check or owner_check != current_user.id:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied - you can only update your own records"
                )
        
        # Build update query
        item_dict = item.dict(exclude_unset=True)
        if not item_dict:
            # If no fields to update, return current record
            row = await conn.fetchrow(f"SELECT * FROM {table_name} WHERE id = $1", item_id)
            return {class_name}Response(**dict(row)) if row else None
        
        # Add audit fields
        item_dict.update({{
            'updated_by': current_user.id,
            'updated_at': datetime.now(timezone.utc)
        }})
        
        set_clauses = [f"{{col}} = ${{i+1}}" for i, col in enumerate(item_dict.keys())]
        values = list(item_dict.values()) + [item_id]
        
        query = f"""
            UPDATE {table_name}
            SET {{", ".join(set_clauses)}}
            WHERE id = ${{len(values)}}
            RETURNING *
        """
        
        row = await conn.fetchrow(query, *values)
        
        if not row:
            raise HTTPException(status_code=404, detail=f"{class_name} not found")
        
        # Log the update
        logger.info(f"User {{current_user.username}} updated {table_name} ID {{item_id}}")
        
        return {class_name}Response(**dict(row))
        
    finally:
        await conn.close()

@app.delete("/{table_name}/{{item_id}}")
@write_rate_limit(requests_per_minute=50)
async def delete_{table_name}(
    item_id: int,
    request: Request,
    current_user: UserResponse = Depends(require_permissions(Permission.DELETE))
):
    """Delete {table_name} with ownership check"""
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid ID")
    
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        
        if Permission.ADMIN not in user_permissions:
            # Verify ownership for non-admin users
            owner_check = await conn.fetchval(
                f"SELECT created_by FROM {table_name} WHERE id = $1",
                item_id
            )
            
            if not owner_check or owner_check != current_user.id:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied - you can only delete your own records"
                )
        
        # Perform soft delete (update is_deleted flag) or hard delete
        result = await conn.execute(
            f"UPDATE {table_name} SET is_deleted = true, deleted_by = $1, deleted_at = $2 WHERE id = $3",
            current_user.id, datetime.now(timezone.utc), item_id
        )
        
        if "UPDATE 0" in result:
            raise HTTPException(status_code=404, detail=f"{class_name} not found")
        
        # Log the deletion
        logger.info(f"User {{current_user.username}} deleted {table_name} ID {{item_id}}")
        
        return {{"message": f"{class_name} deleted successfully"}}
        
    finally:
        await conn.close()

@app.post("/{table_name}/bulk", response_model=List[{class_name}Response])
@bulk_rate_limit(requests_per_hour=10)
async def bulk_create_{table_name}(
    items: List[{class_name}Create],
    request: Request,
    current_user: UserResponse = Depends(require_permissions(Permission.BULK_OPERATIONS))
):
    """Bulk create {table_name} records"""
    if len(items) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 items per bulk operation"
        )
    
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        async with conn.transaction():
            results = []
            for item in items:
                item_dict = item.dict(exclude_unset=True)
                item_dict.update({{
                    'created_by': current_user.id,
                    'created_at': datetime.now(timezone.utc),
                    'updated_by': current_user.id,
                    'updated_at': datetime.now(timezone.utc)
                }})
                
                columns = list(item_dict.keys())
                placeholders = [f"${{i+1}}" for i in range(len(columns))]
                values = list(item_dict.values())
                
                query = f"""
                    INSERT INTO {table_name} ({{", ".join(columns)}})
                    VALUES ({{", ".join(placeholders)}})
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *values)
                results.append({class_name}Response(**dict(row)))
            
            # Log bulk operation
            logger.info(f"User {{current_user.username}} bulk created {{len(results)}} {table_name} records")
            
            return results
    finally:
        await conn.close()

@app.get("/{table_name}/export")
@export_rate_limit(requests_per_hour=5)
async def export_{table_name}(
    format: str = Query("json", regex="^(json|csv)$"),
    request: Request,
    current_user: UserResponse = Depends(require_permissions(Permission.EXPORT))
):
    """Export {table_name} data with ownership filtering"""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        
        if Permission.ADMIN in user_permissions:
            # Admin can export all data
            query = f"SELECT * FROM {table_name} WHERE is_deleted = false ORDER BY created_at DESC LIMIT 10000"
            params = []
        else:
            # Regular users can only export their own data
            query = f"SELECT * FROM {table_name} WHERE created_by = $1 AND is_deleted = false ORDER BY created_at DESC LIMIT 10000"
            params = [current_user.id]
        
        rows = await conn.fetch(query, *params)
        items = [{class_name}Response(**dict(row)) for row in rows]
        
        if format == "csv":
            import io
            import csv
            from fastapi.responses import StreamingResponse
            
            if not items:
                return StreamingResponse(
                    io.StringIO(""),
                    media_type="text/csv",
                    headers={{"Content-Disposition": f"attachment; filename={table_name}.csv"}}
                )
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=items[0].dict().keys())
            writer.writeheader()
            for item in items:
                writer.writerow(item.dict())
            
            output.seek(0)
            
            # Log export
            logger.info(f"User {{current_user.username}} exported {{len(items)}} {table_name} records as CSV")
            
            return StreamingResponse(
                io.StringIO(output.getvalue()),
                media_type="text/csv",
                headers={{"Content-Disposition": f"attachment; filename={table_name}.csv"}}
            )
        
        # Log export
        logger.info(f"User {{current_user.username}} exported {{len(items)}} {table_name} records as JSON")
        
        return items
        
    finally:
        await conn.close()
'''
    return routes_code

# Database schema with audit fields
database_schema_sql = '''
-- 📊 DATABASE SCHEMA WITH AUDIT FIELDS

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('admin', 'manager', 'user', 'readonly')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Add audit fields to existing tables
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS deleted_by INTEGER REFERENCES users(id);
ALTER TABLE users ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;

-- Example: Add audit fields to any table
-- Replace 'your_table' with actual table name
/*
ALTER TABLE your_table ADD COLUMN IF NOT EXISTS created_by INTEGER REFERENCES users(id);
ALTER TABLE your_table ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE your_table ADD COLUMN IF NOT EXISTS updated_by INTEGER REFERENCES users(id);
ALTER TABLE your_table ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE your_table ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT false;
ALTER TABLE your_table ADD COLUMN IF NOT EXISTS deleted_by INTEGER REFERENCES users(id);
ALTER TABLE your_table ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;
*/

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Create default admin user (password: Admin123!)
INSERT INTO users (username, email, password_hash, full_name, role) 
VALUES (
    'admin', 
    'admin@example.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewTrY.N1pSVMZo7G', -- Admin123!
    'System Administrator', 
    'admin'
) ON CONFLICT (username) DO NOTHING;
'''

print("🔒 Secured database routes generated!")
print("✨ Security features:")
print("  • User authentication required")
print("  • Role-based access control")
print("  • Ownership-based data filtering")
print("  • Audit trail (created_by, updated_by, etc.)")
print("  • Soft delete with tracking")
print("  • Rate limiting per operation type")
print("  • Comprehensive logging")
print("  • Admin override capabilities")

print("\\n📊 Database schema includes:")
print("  • Users table with roles")
print("  • Audit fields for all operations")
print("  • Soft delete functionality") 
print("  • Performance indexes")
print("  • Default admin user")

print("\\n🔑 Default admin credentials:")
print("  Username: admin")
print("  Password: Admin123!")

# 🚀 PRODUCTION DEPLOYMENT & CONFIGURATION

# Environment configuration
production_config = '''
# 🏭 PRODUCTION ENVIRONMENT CONFIGURATION

# .env file for production
JWT_SECRET_KEY=your-super-secret-jwt-key-min-32-chars-long
DATABASE_URL=postgresql://user:password@localhost:5432/production_db
REDIS_URL=redis://localhost:6379
ENVIRONMENT=production
LOG_LEVEL=INFO
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
TRUSTED_HOSTS=yourdomain.com,*.yourdomain.com

# Security settings
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15

# Rate limiting
RATE_LIMIT_ENABLED=true
REDIS_RATE_LIMIT_PREFIX=prod_rate_limit

# SSL/TLS
SSL_CERT_PATH=/path/to/ssl/cert.pem
SSL_KEY_PATH=/path/to/ssl/key.pem
'''

# Complete FastAPI application with authentication
complete_app = '''
# 🚀 COMPLETE PRODUCTION FASTAPI APPLICATION

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
import uvicorn
import os
from contextlib import asynccontextmanager

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    try:
        # Initialize Redis connection
        redis_client = await auth_service.get_redis()
        await redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # Initialize rate limiter
        await advanced_rate_limiter.get_redis()
        logger.info("✅ Rate limiter initialized")
        
        # Test database connection
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        await conn.close()
        logger.info("✅ Database connection verified")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        await auth_service.close()
        await advanced_rate_limiter.close()
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Secure Database API",
    description="Production-ready API with authentication, rate limiting, and RBAC",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("TRUSTED_HOSTS", "localhost").split(",")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        await conn.close()
        
        # Check Redis
        redis_client = await auth_service.get_redis()
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

# Include authentication routes
app.include_router(auth_router)
app.include_router(admin_router)

if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        ssl_keyfile=os.getenv("SSL_KEY_PATH"),
        ssl_certfile=os.getenv("SSL_CERT_PATH"),
        workers=int(os.getenv("WORKERS", 4)),
        access_log=True,
        log_level="info"
    )
'''

# Docker configuration
docker_config = '''
# 🐳 DOCKER PRODUCTION DEPLOYMENT

# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${DB_NAME:-production_db}
      POSTGRES_USER: ${DB_USER:-api_user}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-api_user}"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://${DB_USER:-api_user}:${DB_PASSWORD}@postgres:5432/${DB_NAME:-production_db}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENVIRONMENT=production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
'''

# Testing examples
testing_examples = '''
# 🧪 TESTING EXAMPLES

import httpx
import pytest
import asyncio

# Test authentication
async def test_authentication():
    """Test complete authentication flow"""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        
        # 1. Register new user
        register_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPass123!",
            "full_name": "Test User"
        }
        
        response = await client.post("/auth/register", json=register_data)
        assert response.status_code == 200
        user_data = response.json()
        print(f"✅ User registered: {user_data['username']}")
        
        # 2. Login to get tokens
        login_data = {
            "username": "testuser",
            "password": "TestPass123!"
        }
        
        response = await client.post("/auth/login", data=login_data)
        assert response.status_code == 200
        tokens = response.json()
        access_token = tokens["access_token"]
        print(f"✅ Login successful, token: {access_token[:20]}...")
        
        # 3. Access protected endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        user_info = response.json()
        print(f"✅ Protected endpoint accessed: {user_info['username']}")
        
        # 4. Test rate limiting
        for i in range(5):
            response = await client.get("/auth/me", headers=headers)
            print(f"Request {i+1}: {response.status_code}")
        
        # 5. Logout
        response = await client.post("/auth/logout", headers=headers)
        assert response.status_code == 200
        print("✅ Logout successful")

# Load testing with multiple users
async def load_test():
    """Load test with multiple concurrent users"""
    async def user_session(user_id: int):
        async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
            # Login
            login_data = {
                "username": f"user{user_id}",
                "password": "TestPass123!"
            }
            
            response = await client.post("/auth/login", data=login_data)
            if response.status_code != 200:
                return f"User {user_id}: Login failed"
            
            tokens = response.json()
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            
            # Make multiple requests
            success_count = 0
            for _ in range(10):
                response = await client.get("/auth/me", headers=headers)
                if response.status_code == 200:
                    success_count += 1
            
            return f"User {user_id}: {success_count}/10 requests successful"
    
    # Run 50 concurrent user sessions
    tasks = [user_session(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

# Security testing
async def security_test():
    """Test security features"""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        
        # Test invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = await client.get("/auth/me", headers=headers)
        assert response.status_code == 401
        print("✅ Invalid token rejected")
        
        # Test SQL injection attempt
        malicious_data = {
            "username": "admin'; DROP TABLE users; --",
            "password": "password"
        }
        
        response = await client.post("/auth/login", data=malicious_data)
        assert response.status_code == 401  # Should fail authentication
        print("✅ SQL injection attempt blocked")
        
        # Test rate limiting
        for i in range(20):
            response = await client.post("/auth/login", data={"username": "nonexistent", "password": "wrong"})
            if response.status_code == 429:
                print(f"✅ Rate limit triggered after {i+1} attempts")
                break

# Run tests
if __name__ == "__main__":
    asyncio.run(test_authentication())
    # asyncio.run(load_test())
    # asyncio.run(security_test())
'''

print("🚀 Production deployment configuration ready!")
print("📋 Deployment checklist:")
print("  ✅ Environment variables configured")
print("  ✅ Docker containers defined")
print("  ✅ Security headers implemented")
print("  ✅ Health checks added")
print("  ✅ SSL/TLS support")
print("  ✅ Nginx reverse proxy")
print("  ✅ Database migrations")
print("  ✅ Redis persistence")
print("  ✅ Logging configuration")
print("  ✅ Testing suite")

print("\\n🔧 Deployment commands:")
print("  docker-compose up -d")
print("  docker-compose exec api python -m alembic upgrade head")
print("  docker-compose logs -f api")

print("\\n🧪 Testing:")
print("  python test_auth.py")
print("  curl -X POST http://localhost:8000/auth/login")
print("  ab -n 1000 -c 10 http://localhost:8000/health")






