# Database

## Learning Objectives
- [General Database Concepts](#general-database-concepts)
  - [Relational Databases](#relational-databases)
  - [Non-Relational Databases](#non-relational-databases)
  - [SQL vs NoSQL](#sql-vs-nosql)
- [SQL Database Operations](#sql-database-operations)
  - [Creating Tables with Constraints](#creating-tables-with-constraints)
  - [Query Optimization and Indexing](#query-optimization-and-indexing)
  - [Stored Procedures and Functions](#stored-procedures-and-functions)
  - [Views](#views)
  - [Triggers](#triggers)
  - [ACID Properties](#acid-properties)
- [NoSQL Concepts](#nosql-concepts)
  - [Document Storage](#document-storage)
  - [NoSQL Database Types](#nosql-database-types)
  - [Benefits of NoSQL](#benefits-of-nosql)
  - [NoSQL Operations](#nosql-operations)
- [MongoDB Basics](#mongodb-basics)

## General Database Concepts

### Relational Databases
A relational database is a type of database that stores and organizes data in tables (relations) with rows and columns. Data in these tables are related to each other through common fields, enabling complex queries and data relationships.

Key features:
- Structured data format
- Predefined schema
- Table-based
- Uses SQL for querying
- Ensures data integrity through ACID properties

### Non-Relational Databases
Non-relational databases (NoSQL) are databases that store data in a format other than traditional tables. They are designed to handle unstructured data and provide more flexibility in data organization.

Key features:
- Schema-less design
- Horizontal scalability
- Various data models
- Flexible data structure
- Eventually consistent

### SQL vs NoSQL

| Feature | SQL | NoSQL |
|---------|-----|--------|
| Schema | Fixed | Flexible |
| Scaling | Vertical | Horizontal |
| Data Model | Table-based | Various (Document, Key-value, etc.) |
| ACID Compliance | Yes | Varies |
| Query Language | SQL | Database specific |

## SQL Database Operations

### Creating Tables with Constraints
```sql
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    age INT CHECK (age >= 18),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(id)
);
```

Common constraints:
- PRIMARY KEY
- FOREIGN KEY
- UNIQUE
- NOT NULL
- CHECK
- DEFAULT

### Query Optimization and Indexing
Indexes improve query performance by creating data structures that speed up data retrieval.

Creating an index:
```sql
CREATE INDEX idx_username ON users(username);
CREATE UNIQUE INDEX idx_email ON users(email);
```

Best practices:
- Index frequently queried columns
- Avoid over-indexing (impacts INSERT/UPDATE performance)
- Use EXPLAIN to analyze query performance
- Consider compound indexes for multiple columns

### Stored Procedures and Functions
Stored procedures are prepared SQL statements that can be reused.

Example stored procedure:
```sql
DELIMITER //
CREATE PROCEDURE GetUsersByAge(IN min_age INT)
BEGIN
    SELECT * FROM users WHERE age >= min_age;
END //
DELIMITER ;

-- Usage
CALL GetUsersByAge(21);
```

Example function:
```sql
DELIMITER //
CREATE FUNCTION CalculateAge(birth_date DATE)
RETURNS INT
DETERMINISTIC
BEGIN
    RETURN YEAR(CURRENT_DATE) - YEAR(birth_date);
END //
DELIMITER ;
```

### Views
Views are virtual tables based on the result set of an SQL statement.

```sql
CREATE VIEW active_users AS
SELECT username, email
FROM users
WHERE last_login >= DATE_SUB(NOW(), INTERVAL 30 DAY);

-- Using the view
SELECT * FROM active_users;
```

### Triggers
Triggers are special procedures that automatically run when specific database events occur.

```sql
DELIMITER //
CREATE TRIGGER before_user_update
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    SET NEW.updated_at = NOW();
END //
DELIMITER ;
```

### ACID Properties
ACID ensures database transactions are processed reliably:
- **Atomicity**: Transactions are all-or-nothing
- **Consistency**: Database remains in a valid state
- **Isolation**: Transactions don't interfere with each other
- **Durability**: Completed transactions are permanent

## NoSQL Concepts

### Document Storage
Document storage is a type of NoSQL database that stores data in flexible, JSON-like documents. Each document can have different fields, and the structure can be changed over time.

### NoSQL Database Types
1. **Document Stores** (MongoDB, CouchDB)
   - Store data in document format
   - Flexible schema
   - Good for nested data

2. **Key-Value Stores** (Redis, DynamoDB)
   - Simple key-value pairs
   - High performance
   - Good for caching

3. **Column-Family Stores** (Cassandra)
   - Stores data in columns
   - Good for large-scale data

4. **Graph Databases** (Neo4j)
   - Store data in nodes and edges
   - Good for connected data

### Benefits of NoSQL
- Flexible schema
- Horizontal scalability
- Better performance for specific use cases
- Handles unstructured data well
- Simpler development in some cases

### NoSQL Operations
Basic operations in NoSQL databases:

```javascript
// Create
db.collection.insertOne({})
db.collection.insertMany([])

// Read
db.collection.find({})
db.collection.findOne({})

// Update
db.collection.updateOne({})
db.collection.updateMany({})

// Delete
db.collection.deleteOne({})
db.collection.deleteMany({})
```

## MongoDB Basics

### Installation
```bash
# Download and install MongoDB
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-5.0.5.tgz
tar -xzf mongodb-linux-x86_64-ubuntu2004-5.0.5.tgz
```

### Basic Operations
```javascript
// Connect to MongoDB
mongosh "mongodb://localhost:27017"

// Create database
use myDatabase

// Insert document
db.users.insertOne({
    name: "John Doe",
    age: 30,
    email: "john@example.com"
})

// Find documents
db.users.find({ age: { $gte: 25 } })

// Update document
db.users.updateOne(
    { name: "John Doe" },
    { $set: { age: 31 } }
)

// Delete document
db.users.deleteOne({ name: "John Doe" })
```

### Advanced Queries
```javascript
// Aggregation
db.orders.aggregate([
    { $match: { status: "completed" } },
    { $group: { _id: "$userId", total: { $sum: "$amount" } } }
])

// Indexing
db.users.createIndex({ email: 1 })

// Text search
db.articles.createIndex({ content: "text" })
db.articles.find({ $text: { $search: "mongodb" } })
```