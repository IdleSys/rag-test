# Hakim Backend

A Django REST API backend for AI assistant management with user authentication, CORS support, and PostgreSQL database.

## 🚀 Features

- **User Management**: Custom user model with email/username authentication
- **AI Assistants**: Create and manage AI assistants with different models
- **JWT Authentication**: Secure token-based authentication
- **CORS Support**: Configured for frontend integration
- **Categories**: Organize assistants by categories
- **Chat System**: Ready for chat functionality
- **Admin Interface**: Django admin for easy management

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 12+
- Git

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd hakim-real-backend
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Django Settings
DEBUG=True
SECRET_KEY=django-insecure-dd(3v9vxgv)#qqfynzyezdu14r60=jakje#*i4gp$fjb^!)krw
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Settings
DB_NAME=hakim_real_db
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=5432

# CORS Settings
CORS_ALLOWED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080,http://localhost:3000,http://127.0.0.1:3000

# JWT Settings
JWT_ACCESS_TOKEN_LIFETIME_MINUTES=60
JWT_REFRESH_TOKEN_LIFETIME_DAYS=7
```

**Note**: Copy from `.env.example` and update the values:
```bash
cp .env.example .env
```

### 5. Database Setup

#### Option A: PostgreSQL (Recommended)
1. Install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/)
2. Create database:
```sql
-- Connect to PostgreSQL as superuser
CREATE DATABASE hakim_real_db;
CREATE USER postgres WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE hakim_real_db TO postgres;
```
3. Update `.env` with your database credentials

#### Option B: SQLite (Development Only)
If you prefer SQLite for development, update `core/settings.py`:
```python
# Comment out PostgreSQL config and use:
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

### 6. Run Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Create Superuser
```bash
python manage.py createsuperuser
```

### 8. Start Development Server
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
hakim-real-backend/
├── core/                   # Django project settings
│   ├── settings.py        # Main settings
│   ├── urls.py           # URL configuration
│   └── wsgi.py           # WSGI config
├── users/                 # User management app
│   ├── models.py         # Custom User model
│   ├── views.py          # Auth views
│   ├── serializers.py    # User serializers
│   └── urls.py           # User endpoints
├── assistants/            # AI assistants app
│   ├── models.py         # Assistant & Category models
│   ├── admin.py          # Admin interface
│   └── apps.py           # App configuration
├── chats/                 # Chat functionality
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── manage.py             # Django management
```

## 🔗 API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/signup` - User registration
- `POST /api/v1/auth/refresh` - Refresh JWT token
- `POST /api/v1/auth/logout` - User logout

### User Management
- `GET /api/v1/profile` - Get user profile
- `PUT /api/v1/profile/update` - Update profile
- `POST /api/v1/profile/change-password` - Change password
- `GET /api/v1/users` - List users
- `GET /api/v1/users/<id>` - Get user details

### API Overview
- `GET /api/v1/` - API documentation overview

## 🔧 Development

### Running Tests
```bash
python manage.py test
```

### Creating New Apps
```bash
python manage.py startapp app_name
```

### Making Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```
