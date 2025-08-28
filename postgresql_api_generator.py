# FastAPI PostgreSQL CRUD Generator
# This will automatically generate CRUD APIs based on table schemas

import asyncio
import asyncpg
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# First, let's create a table schema inspector
@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[str]
    is_primary_key: bool

@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    primary_keys: List[str]

class PostgreSQLSchemaInspector:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def get_all_tables(self) -> List[str]:
        """Get all table names from the database"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
            rows = await conn.fetch(query)
            return [row['table_name'] for row in rows]
        finally:
            await conn.close()
    
    async def get_table_schema(self, table_name: str) -> TableInfo:
        """Get detailed schema information for a specific table"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            # Get column information
            column_query = """
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                ON tc.constraint_name = ku.constraint_name
                WHERE tc.table_name = $1 AND tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.column_name = pk.column_name
            WHERE c.table_name = $1
            ORDER BY c.ordinal_position;
            """
            
            rows = await conn.fetch(column_query, table_name)
            columns = []
            primary_keys = []
            
            for row in rows:
                col = ColumnInfo(
                    name=row['column_name'],
                    data_type=row['data_type'],
                    is_nullable=row['is_nullable'] == 'YES',
                    default_value=row['column_default'],
                    is_primary_key=row['is_primary_key']
                )
                columns.append(col)
                if col.is_primary_key:
                    primary_keys.append(col.name)
            
            return TableInfo(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys
            )
        finally:
            await conn.close()

# Example usage
async def demo_schema_inspection():
    # Replace with your actual connection string
    connection_string = "postgresql://user:password@localhost:5432/dbname"
    
    inspector = PostgreSQLSchemaInspector(connection_string)
    
    try:
        # Get all tables
        tables = await inspector.get_all_tables()
        print(f"Found {len(tables)} tables:")
        for table in tables[:3]:  # Show first 3 tables
            print(f"  - {table}")
            
        # Get schema for first table (example)
        if tables:
            schema = await inspector.get_table_schema(tables[0])
            print(f"\nSchema for '{schema.name}':")
            for col in schema.columns:
                pk_marker = " (PK)" if col.is_primary_key else ""
                nullable = " NULL" if col.is_nullable else " NOT NULL"
                print(f"  {col.name}: {col.data_type}{nullable}{pk_marker}")
                
    except Exception as e:
        print(f"Demo error (expected if no database): {e}")

# Run the demo
# asyncio.run(demo_schema_inspection())


# FastAPI Code Generator for PostgreSQL Tables

class FastAPICodeGenerator:
    def __init__(self):
        self.type_mapping = {
            'integer': 'int',
            'bigint': 'int', 
            'smallint': 'int',
            'numeric': 'float',
            'real': 'float',
            'double precision': 'float',
            'text': 'str',
            'character varying': 'str',
            'character': 'str',
            'boolean': 'bool',
            'date': 'datetime.date',
            'timestamp without time zone': 'datetime.datetime',
            'timestamp with time zone': 'datetime.datetime',
            'time': 'datetime.time',
            'uuid': 'str',
            'json': 'dict',
            'jsonb': 'dict'
        }
    
    def generate_pydantic_model(self, table_info: TableInfo) -> str:
        """Generate Pydantic models for a table"""
        class_name = self._to_camel_case(table_info.name)
        
        # Base model (for creation)
        create_fields = []
        # Response model (for reading)
        response_fields = []
        # Update model (for updates - all optional)
        update_fields = []
        
        for col in table_info.columns:
            python_type = self.type_mapping.get(col.data_type, 'str')
            
            # For create model - exclude auto-generated primary keys
            if not (col.is_primary_key and col.default_value and 'nextval' in str(col.default_value)):
                if col.is_nullable:
                    create_fields.append(f"    {col.name}: Optional[{python_type}] = None")
                else:
                    create_fields.append(f"    {col.name}: {python_type}")
            
            # For response model - include all fields
            if col.is_nullable:
                response_fields.append(f"    {col.name}: Optional[{python_type}] = None")
            else:
                response_fields.append(f"    {col.name}: {python_type}")
            
            # For update model - all fields optional
            update_fields.append(f"    {col.name}: Optional[{python_type}] = None")
        
        models = f'''
# Pydantic models for {table_info.name}
class {class_name}Create(BaseModel):
{chr(10).join(create_fields) if create_fields else "    pass"}

class {class_name}Update(BaseModel):
{chr(10).join(update_fields)}

class {class_name}Response(BaseModel):
{chr(10).join(response_fields)}
    
    class Config:
        from_attributes = True
'''
        return models
    
    def generate_crud_operations(self, table_info: TableInfo) -> str:
        """Generate CRUD operations for a table"""
        class_name = self._to_camel_case(table_info.name)
        table_name = table_info.name
        primary_key = table_info.primary_keys[0] if table_info.primary_keys else 'id'
        
        # Generate column lists
        all_columns = [col.name for col in table_info.columns]
        insert_columns = [col.name for col in table_info.columns 
                         if not (col.is_primary_key and col.default_value and 'nextval' in str(col.default_value))]
        
        crud_code = f'''
# CRUD operations for {table_name}
class {class_name}CRUD:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def create(self, item: {class_name}Create) -> {class_name}Response:
        conn = await asyncpg.connect(self.connection_string)
        try:
            # Build dynamic insert query
            item_dict = item.dict(exclude_unset=True)
            columns = list(item_dict.keys())
            placeholders = [f"${i+1}" for i in range(len(columns))]
            values = list(item_dict.values())
            
            query = f"""
            INSERT INTO {table_name} ({", ".join(columns)})
            VALUES ({", ".join(placeholders)})
            RETURNING *
            """
            
            row = await conn.fetchrow(query, *values)
            return {class_name}Response(**dict(row))
        finally:
            await conn.close()
    
    async def get_by_id(self, {primary_key}: int) -> Optional[{class_name}Response]:
        conn = await asyncpg.connect(self.connection_string)
        try:
            query = f"SELECT * FROM {table_name} WHERE {primary_key} = $1"
            row = await conn.fetchrow(query, {primary_key})
            return {class_name}Response(**dict(row)) if row else None
        finally:
            await conn.close()
    
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[{class_name}Response]:
        conn = await asyncpg.connect(self.connection_string)
        try:
            query = f"SELECT * FROM {table_name} OFFSET $1 LIMIT $2"
            rows = await conn.fetch(query, skip, limit)
            return [{class_name}Response(**dict(row)) for row in rows]
        finally:
            await conn.close()
    
    async def update(self, {primary_key}: int, item: {class_name}Update) -> Optional[{class_name}Response]:
        conn = await asyncpg.connect(self.connection_string)
        try:
            # Build dynamic update query
            item_dict = item.dict(exclude_unset=True)
            if not item_dict:
                return await self.get_by_id({primary_key})
            
            set_clauses = [f"{{col}} = ${i+1}" for i, col in enumerate(item_dict.keys(), 1)]
            values = list(item_dict.values()) + [{primary_key}]
            
            query = f"""
            UPDATE {table_name} 
            SET {", ".join(set_clauses)}
            WHERE {primary_key} = ${len(values)}
            RETURNING *
            """
            
            row = await conn.fetchrow(query, *values)
            return {class_name}Response(**dict(row)) if row else None
        finally:
            await conn.close()
    
    async def delete(self, {primary_key}: int) -> bool:
        conn = await asyncpg.connect(self.connection_string)
        try:
            query = f"DELETE FROM {table_name} WHERE {primary_key} = $1"
            result = await conn.execute(query, {primary_key})
            return "DELETE 1" in result
        finally:
            await conn.close()
'''
        return crud_code
    
    def generate_api_routes(self, table_info: TableInfo) -> str:
        """Generate FastAPI routes for a table"""
        class_name = self._to_camel_case(table_info.name)
        table_name = table_info.name
        primary_key = table_info.primary_keys[0] if table_info.primary_keys else 'id'
        
        routes_code = f'''
# FastAPI routes for {table_name}
@app.post("/{table_name}/", response_model={class_name}Response)
async def create_{table_name}(item: {class_name}Create):
    """Create a new {table_name} record"""
    crud = {class_name}CRUD(DATABASE_URL)
    return await crud.create(item)

@app.get("/{table_name}/{{item_id}}", response_model={class_name}Response)
async def get_{table_name}(item_id: int):
    """Get a {table_name} record by ID"""
    crud = {class_name}CRUD(DATABASE_URL)
    item = await crud.get_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"{class_name} not found")
    return item

@app.get("/{table_name}/", response_model=List[{class_name}Response])
async def list_{table_name}(skip: int = 0, limit: int = 100):
    """List {table_name} records with pagination"""
    crud = {class_name}CRUD(DATABASE_URL)
    return await crud.get_all(skip=skip, limit=limit)

@app.put("/{table_name}/{{item_id}}", response_model={class_name}Response)
async def update_{table_name}(item_id: int, item: {class_name}Update):
    """Update a {table_name} record"""
    crud = {class_name}CRUD(DATABASE_URL)
    updated_item = await crud.update(item_id, item)
    if not updated_item:
        raise HTTPException(status_code=404, detail=f"{class_name} not found")
    return updated_item

@app.delete("/{table_name}/{{item_id}}")
async def delete_{table_name}(item_id: int):
    """Delete a {table_name} record"""
    crud = {class_name}CRUD(DATABASE_URL)
    success = await crud.delete(item_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"{class_name} not found")
    return {{"message": f"{class_name} deleted successfully"}}
'''
        return routes_code
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase"""
        components = snake_str.split('_')
        return ''.join(word.capitalize() for word in components)

# Example usage
def demo_code_generation():
    # Create sample table info
    sample_columns = [
        ColumnInfo("id", "integer", False, "nextval('users_id_seq'::regclass)", True),
        ColumnInfo("email", "character varying", False, None, False),
        ColumnInfo("name", "character varying", True, None, False),
        ColumnInfo("created_at", "timestamp without time zone", False, "now()", False),
        ColumnInfo("is_active", "boolean", False, "true", False),
    ]
    
    sample_table = TableInfo("users", sample_columns, ["id"])
    
    generator = FastAPICodeGenerator()
    
    print("=== GENERATED PYDANTIC MODELS ===")
    print(generator.generate_pydantic_model(sample_table))
    
    print("\n=== GENERATED CRUD OPERATIONS ===")
    print(generator.generate_crud_operations(sample_table)[:500] + "...")
    
    print("\n=== GENERATED API ROUTES ===")
    print(generator.generate_api_routes(sample_table)[:500] + "...")

demo_code_generation()


# Complete FastAPI Application Generator

class FastAPIAppGenerator:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.schema_inspector = PostgreSQLSchemaInspector()
        self.code_generator = FastAPICodeGenerator()
    
    async def generate_complete_app(self, output_file: str = "generated_app.py"):
        """Generate a complete FastAPI application for all tables in the database"""
        
        # Get all tables
        tables = await self.schema_inspector.get_all_tables(self.database_url)
        
        # Generate application header
        app_code = self._generate_app_header()
        
        # Generate models, CRUD, and routes for each table
        for table_name in tables:
            table_info = await self.schema_inspector.get_table_schema(self.database_url, table_name)
            
            app_code += f"\n# ==================== {table_name.upper()} ====================\n"
            app_code += self.code_generator.generate_pydantic_model(table_info)
            app_code += self.code_generator.generate_crud_operations(table_info)
            app_code += self.code_generator.generate_api_routes(table_info)
        
        # Add application footer
        app_code += self._generate_app_footer()
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(app_code)
        
        print(f"✅ Generated complete FastAPI application: {output_file}")
        print(f"📊 Generated CRUD operations for {len(tables)} tables")
        return output_file
    
    def _generate_app_header(self) -> str:
        """Generate the FastAPI application header with imports and setup"""
        return f'''"""
Auto-generated FastAPI application for PostgreSQL database
Generated with FastAPIAppGenerator
"""

import asyncpg
import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

# Database configuration
DATABASE_URL = "{self.database_url}"

# FastAPI app initialization
app = FastAPI(
    title="Auto-Generated PostgreSQL API",
    description="Automatically generated CRUD API for PostgreSQL tables",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.close()
        return {{"status": "healthy", "database": "connected"}}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {{str(e)}}")

# Database connection helper
async def get_db_connection():
    """Get database connection"""
    return await asyncpg.connect(DATABASE_URL)

'''
    
    def _generate_app_footer(self) -> str:
        """Generate the FastAPI application footer with startup code"""
        return '''

# Application startup
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the auto-generated FastAPI application")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting FastAPI application...")
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔧 ReDoc: http://{args.host}:{args.port}/redoc")
    
    uvicorn.run(
        "generated_app:app" if not args.reload else "__main__:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
'''

# Enhanced demo with table discovery
async def generate_api_for_database(database_url: str):
    """Complete example: Generate API for entire database"""
    
    try:
        generator = FastAPIAppGenerator(database_url)
        
        # Generate the complete application
        output_file = await generator.generate_complete_app("my_database_api.py")
        
        print(f"\n✨ FastAPI application generated successfully!")
        print(f"📁 File: {output_file}")
        print(f"\n🏃‍♂️ To run the application:")
        print(f"python {output_file}")
        print(f"\n🌐 API will be available at:")
        print(f"• Main API: http://localhost:8000")
        print(f"• Interactive docs: http://localhost:8000/docs")
        print(f"• ReDoc: http://localhost:8000/redoc")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error generating API: {e}")
        return None

# Example usage (modify the database URL for your setup)
DATABASE_URL = "postgresql://username:password@localhost:5432/database_name"

print("🔧 FastAPI PostgreSQL Generator Ready!")
print(f"📚 Usage example:")
print(f"await generate_api_for_database('{DATABASE_URL}')")

# Uncomment the line below to run the generator
# await generate_api_for_database(DATABASE_URL)


# Advanced Features for FastAPI PostgreSQL Generator

class AdvancedFastAPIGenerator(FastAPIAppGenerator):
    """Extended generator with advanced features"""
    
    def generate_advanced_crud_operations(self, table_info: TableInfo) -> str:
        """Generate CRUD operations with filtering, sorting, and bulk operations"""
        class_name = self._to_camel_case(table_info.name)
        table_name = table_info.name
        primary_key = table_info.primary_keys[0] if table_info.primary_keys else 'id'
        
        # Get filterable columns (non-blob types)
        filterable_columns = [col.name for col in table_info.columns 
                            if col.data_type not in ['bytea', 'json', 'jsonb']]
        
        crud_code = f'''
# Advanced CRUD operations for {table_name}
class {class_name}AdvancedCRUD({class_name}CRUD):
    
    async def get_with_filters(self, 
                             skip: int = 0, 
                             limit: int = 100,
                             filters: Dict[str, Any] = None,
                             sort_by: str = None,
                             sort_desc: bool = False) -> List[{class_name}Response]:
        """Get records with filtering and sorting"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            # Build base query
            query = f"SELECT * FROM {table_name}"
            params = []
            param_count = 0
            
            # Add filters
            if filters:
                where_clauses = []
                for column, value in filters.items():
                    if column in {filterable_columns} and value is not None:
                        param_count += 1
                        if isinstance(value, str) and '%' in value:
                            where_clauses.append(f"{{column}} ILIKE ${param_count}")
                        else:
                            where_clauses.append(f"{{column}} = ${param_count}")
                        params.append(value)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            # Add sorting
            if sort_by and sort_by in {filterable_columns}:
                order_direction = "DESC" if sort_desc else "ASC"
                query += f" ORDER BY {{sort_by}} {{order_direction}}"
            
            # Add pagination
            param_count += 1
            params.append(skip)
            param_count += 1
            params.append(limit)
            query += f" OFFSET ${param_count-1} LIMIT ${param_count}"
            
            rows = await conn.fetch(query, *params)
            return [{class_name}Response(**dict(row)) for row in rows]
        finally:
            await conn.close()
    
    async def count_with_filters(self, filters: Dict[str, Any] = None) -> int:
        """Count records with filters"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            params = []
            
            if filters:
                where_clauses = []
                param_count = 0
                for column, value in filters.items():
                    if column in {filterable_columns} and value is not None:
                        param_count += 1
                        if isinstance(value, str) and '%' in value:
                            where_clauses.append(f"{{column}} ILIKE ${param_count}")
                        else:
                            where_clauses.append(f"{{column}} = ${param_count}")
                        params.append(value)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            result = await conn.fetchval(query, *params)
            return result
        finally:
            await conn.close()
    
    async def bulk_create(self, items: List[{class_name}Create]) -> List[{class_name}Response]:
        """Create multiple records in a single transaction"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            async with conn.transaction():
                results = []
                for item in items:
                    item_dict = item.dict(exclude_unset=True)
                    columns = list(item_dict.keys())
                    placeholders = [f"${i+1}" for i in range(len(columns))]
                    values = list(item_dict.values())
                    
                    query = f"""
                    INSERT INTO {table_name} ({", ".join(columns)})
                    VALUES ({", ".join(placeholders)})
                    RETURNING *
                    """
                    
                    row = await conn.fetchrow(query, *values)
                    results.append({class_name}Response(**dict(row)))
                
                return results
        finally:
            await conn.close()
    
    async def bulk_update(self, updates: Dict[int, {class_name}Update]) -> List[{class_name}Response]:
        """Update multiple records by ID"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            async with conn.transaction():
                results = []
                for record_id, update_data in updates.items():
                    result = await self.update(record_id, update_data)
                    if result:
                        results.append(result)
                return results
        finally:
            await conn.close()
    
    async def bulk_delete(self, ids: List[int]) -> int:
        """Delete multiple records by ID"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            placeholders = [f"${i+1}" for i in range(len(ids))]
            query = f"DELETE FROM {table_name} WHERE {primary_key} IN ({', '.join(placeholders)})"
            result = await conn.execute(query, *ids)
            return int(result.split()[-1])  # Extract number of deleted rows
        finally:
            await conn.close()
'''
        return crud_code
    
    def generate_advanced_api_routes(self, table_info: TableInfo) -> str:
        """Generate API routes with advanced features"""
        class_name = self._to_camel_case(table_info.name)
        table_name = table_info.name
        primary_key = table_info.primary_keys[0] if table_info.primary_keys else 'id'
        
        # Get filterable columns
        filterable_columns = [col.name for col in table_info.columns 
                            if col.data_type not in ['bytea', 'json', 'jsonb']]
        
        routes_code = f'''
# Advanced FastAPI routes for {table_name}

# Filter model
class {class_name}Filter(BaseModel):
    {chr(10).join([f"    {col}: Optional[str] = None" for col in filterable_columns[:10]])}  # Limit to first 10 columns

# Pagination response model
class {class_name}PaginatedResponse(BaseModel):
    items: List[{class_name}Response]
    total: int
    page: int
    per_page: int
    pages: int

@app.get("/{table_name}/search", response_model={class_name}PaginatedResponse)
async def search_{table_name}(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    sort_by: Optional[str] = Query(None),
    sort_desc: bool = Query(False),
    filters: {class_name}Filter = Depends()
):
    """Advanced search with filtering, sorting, and pagination"""
    crud = {class_name}AdvancedCRUD(DATABASE_URL)
    
    skip = (page - 1) * per_page
    filter_dict = {{k: v for k, v in filters.dict().items() if v is not None}}
    
    # Get items and total count
    items = await crud.get_with_filters(
        skip=skip, 
        limit=per_page, 
        filters=filter_dict,
        sort_by=sort_by,
        sort_desc=sort_desc
    )
    total = await crud.count_with_filters(filter_dict)
    
    return {class_name}PaginatedResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page
    )

@app.post("/{table_name}/bulk", response_model=List[{class_name}Response])
async def bulk_create_{table_name}(items: List[{class_name}Create]):
    """Create multiple {table_name} records"""
    crud = {class_name}AdvancedCRUD(DATABASE_URL)
    return await crud.bulk_create(items)

@app.put("/{table_name}/bulk", response_model=List[{class_name}Response])
async def bulk_update_{table_name}(updates: Dict[int, {class_name}Update]):
    """Update multiple {table_name} records"""
    crud = {class_name}AdvancedCRUD(DATABASE_URL)
    return await crud.bulk_update(updates)

@app.delete("/{table_name}/bulk")
async def bulk_delete_{table_name}(ids: List[int]):
    """Delete multiple {table_name} records"""
    crud = {class_name}AdvancedCRUD(DATABASE_URL)
    deleted_count = await crud.bulk_delete(ids)
    return {{"message": f"Deleted {{deleted_count}} {table_name} records"}}

@app.get("/{table_name}/export")
async def export_{table_name}(
    format: str = Query("json", regex="^(json|csv)$"),
    filters: {class_name}Filter = Depends()
):
    """Export {table_name} data in JSON or CSV format"""
    crud = {class_name}AdvancedCRUD(DATABASE_URL)
    
    filter_dict = {{k: v for k, v in filters.dict().items() if v is not None}}
    items = await crud.get_with_filters(filters=filter_dict, limit=10000)  # Large limit for export
    
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
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={{"Content-Disposition": f"attachment; filename={table_name}.csv"}}
        )
    
    return items  # JSON format
'''
        return routes_code

# Example of using the advanced generator
async def generate_advanced_api(database_url: str):
    """Generate API with advanced features"""
    generator = AdvancedFastAPIGenerator(database_url)
    
    # Override the code generation methods to use advanced versions
    original_crud = generator.code_generator.generate_crud_operations
    original_routes = generator.code_generator.generate_api_routes
    
    generator.code_generator.generate_crud_operations = generator.generate_advanced_crud_operations
    generator.code_generator.generate_api_routes = generator.generate_advanced_api_routes
    
    return await generator.generate_complete_app("advanced_api.py")

print("🚀 Advanced FastAPI Generator with:")
print("• Filtering & search")
print("• Sorting & pagination") 
print("• Bulk operations")
print("• Export functionality")
print("• Transaction support")


