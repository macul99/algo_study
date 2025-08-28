import time  
import functools  
import redis  
  
# You can configure Redis connection as per your needs  
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)  
  
class RateLimitException(Exception):  
    pass  
  
def rate_limit(key_func, limit: int, window: int):  
    """  
    Rate limit decorator using Redis.  
  
    Parameters:  
    - key_func: Callable that returns a unique key (e.g., user ID or IP)  
    - limit: Number of allowed requests  
    - window: Time window in seconds  
    """  
    def decorator(func):  
        @functools.wraps(func)  
        def wrapper(*args, **kwargs):  
            key = key_func(*args, **kwargs)  
            redis_key = f"rate-limit:{key}"  
  
            try:  
                # Use Redis INCR + EXPIRE for atomic rate limiting  
                current = redis_client.incr(redis_key)  
                if current == 1:  
                    # Set expiration only on first increment  
                    redis_client.expire(redis_key, window)  
                if current > limit:  
                    raise RateLimitException("Too many requests, slow down.")  
            except redis.exceptions.ConnectionError:  
                # Optional: Fail open or fail closed  
                # Fail open: Allow request if Redis is down  
                pass  
  
            return func(*args, **kwargs)  
        return wrapper  
    return decorator  

def user_key_func(user_id, *args, **kwargs):  
    return f"user:{user_id}"  
  
@rate_limit(user_key_func, limit=5, window=60)  
def my_protected_view(user_id):  
    return f"Hello, user {user_id}!"  
  
# Example usage  
for i in range(7):  
    try:  
        print(my_protected_view(42))  
    except RateLimitException as e:  
        print(e)  

pip install fastapi redis uvicorn  

from fastapi import FastAPI, Request, HTTPException, status, Depends  
import redis.asyncio as redis  
import time  
  
app = FastAPI()  
  
# Configure Redis connection  
redis_client = redis.from_url("redis://localhost:6379/0", decode_responses=True)  
  
async def rate_limiter(  
    request: Request,  
    limit: int = 5,  
    window: int = 60,  
    key_prefix: str = "rl",  
    identifier: str = None,  
    fail_open: bool = True,  
):  
    """  
    FastAPI dependency for rate limiting using Redis.  
  
    Args:  
        request: FastAPI request object.  
        limit: Allowed requests per window (default 5).  
        window: Window in seconds (default 60).  
        key_prefix: Prefix for Redis keys.  
        identifier: Optional unique identifier (e.g., user id); falls back to IP.  
        fail_open: If True, allows requests if Redis is down; else, fails closed.  
    """  
    # Use identifier (e.g., user id, API key) or fallback to client IP  
    key = identifier or request.client.host  
    redis_key = f"{key_prefix}:{key}"  
  
    try:  
        # Use Redis pipeline for atomicity  
        pipe = redis_client.pipeline()  
        pipe.incr(redis_key)  
        pipe.ttl(redis_key)  
        results = await pipe.execute()  
        current = int(results[0])  
        ttl = int(results[1])  
  
        if current == 1 or ttl == -1:  
            # Set expiration only if key is new or has no expiry  
            await redis_client.expire(redis_key, window)  
            reset = window  
        else:  
            reset = ttl  
  
        if current > limit:  
            raise HTTPException(  
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,  
                detail=f"Rate limit exceeded. Try again in {reset} seconds.",  
                headers={  
                    "Retry-After": str(reset),  
                    "X-RateLimit-Limit": str(limit),  
                    "X-RateLimit-Remaining": str(max(0, limit - current)),  
                    "X-RateLimit-Reset": str(int(time.time() + reset)),  
                }  
            )  
        # Optionally, return rate limit info for use in endpoint  
        return {  
            "limit": limit,  
            "remaining": max(0, limit - current),  
            "reset": int(time.time() + reset)  
        }  
    except redis.exceptions.ConnectionError:  
        if fail_open:  
            return None  
        else:  
            raise HTTPException(  
                status_code=500,  
                detail="Rate limiting backend unavailable."  
            )  
# Per-IP Rate Limiting
@app.get("/limited")  
async def limited_endpoint(  
    rate_limit_info: dict = Depends(rate_limiter)  
):  
    return {"message": "You did not hit the rate limit!", "rate_limit": rate_limit_info}  

#Per-User (or API key) Rate Limiting
# Example dependency to extract user id (replace with your actual auth logic)  
async def get_user_id(request: Request):  
    # Replace with real authentication!  
    return request.headers.get("x-user-id", None)  
  
@app.get("/user-limited")  
async def user_limited_endpoint(  
    user_id: str = Depends(get_user_id),  
    rate_limit_info: dict = Depends(lambda request: rate_limiter(request, identifier=user_id))  
):  
    return {"message": f"Hello user {user_id}", "rate_limit": rate_limit_info}  

pip install fastapi[all] "python-jose[cryptography]" passlib[bcrypt] sqlalchemy  

import os  
from datetime import datetime, timedelta  
from jose import JWTError, jwt  
from passlib.context import CryptContext  
  
# Secret key and algorithm (use env var in production!)  
SECRET_KEY = os.getenv("SECRET_KEY", "SUPER_SECRET_KEY")  # Replace in prod!  
ALGORITHM = "HS256"  
ACCESS_TOKEN_EXPIRE_MINUTES = 30  
  
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")  
  
def verify_password(plain, hashed):  
    return pwd_context.verify(plain, hashed)  
  
def get_password_hash(password):  
    return pwd_context.hash(password)  
  
def create_access_token(data: dict, expires_delta: timedelta | None = None):  
    to_encode = data.copy()  
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))  
    to_encode.update({"exp": expire})  
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)  

from pydantic import BaseModel  
  
class User(BaseModel):  
    username: str  
    full_name: str | None = None  
  
class UserInDB(User):  
    hashed_password: str  
  
class UserCreate(BaseModel):  
    username: str  
    password: str  

# Replace this with your SQLAlchemy or ORM logic in prod!  
fake_users_db = {  
    "alice": {  
        "username": "alice",  
        "full_name": "Alice Smith",  
        "hashed_password": get_password_hash("wonderland"),  
    }  
}  
def get_user(username: str) -> UserInDB | None:  
    user = fake_users_db.get(username)  
    if user:  
        return UserInDB(**user)  
    return None  

from fastapi import FastAPI, HTTPException, Depends, status  
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm  
  
app = FastAPI()  
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")  
  
def authenticate_user(username: str, password: str) -> UserInDB | None:  
    user = get_user(username)  
    if not user or not verify_password(password, user.hashed_password):  
        return None  
    return user  
  
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:  
    credentials_exception = HTTPException(  
        status_code=status.HTTP_401_UNAUTHORIZED,  
        detail="Could not validate credentials",  
        headers={"WWW-Authenticate": "Bearer"},  
    )  
    try:  
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  
        username: str = payload.get("sub")  
        if username is None:  
            raise credentials_exception  
    except JWTError:  
        raise credentials_exception  
    user = get_user(username)  
    if user is None:  
        raise credentials_exception  
    return user  

from fastapi.responses import JSONResponse  
  
@app.post("/token")  
async def login(form_data: OAuth2PasswordRequestForm = Depends()):  
    user = authenticate_user(form_data.username, form_data.password)  
    if not user:  
        raise HTTPException(status_code=400, detail="Incorrect username or password")  
    access_token = create_access_token(  
        data={"sub": user.username},  
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)  
    )  
    return {"access_token": access_token, "token_type": "bearer"}  
  
@app.get("/users/me")  
async def read_users_me(current_user: User = Depends(get_current_user)):  
    return current_user  
  
# Example of protected endpoint  
@app.get("/protected")  
async def protected_route(current_user: User = Depends(get_current_user)):  
    return {"msg": f"Hello, {current_user.username}!"}  

