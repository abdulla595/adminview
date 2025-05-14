import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import sqlite3
import hashlib
import jwt
import datetime
import os
from functools import wraps

# JWT secret key - replaced with API URL as requested
SECRET_KEY = 'https://sheet2api.com/v1/e3H9HQWFR1ao/student_data'  # Change this in production

def init_db():
    try:
        # Ensure the database directory exists
        db_dir = os.path.dirname('users.db')
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (username TEXT PRIMARY KEY, password TEXT, role TEXT, 
                    email TEXT, last_login DATETIME)''')
        
        default_password = hashlib.sha256('password123'.encode()).hexdigest()
        default_users = [
            ('admin', default_password, 'admin', 'admin@hct.ac.ae'),
            ('instructor1', default_password, 'instructor', 'instructor1@hct.ac.ae'),
            ('advisor1', default_password, 'advisor', 'advisor1@hct.ac.ae'),
            ('student1', default_password, 'student', 'student1@hct.ac.ae')
        ]
        
        for user in default_users:
            try:
                c.execute('INSERT INTO users (username, password, role, email) VALUES (?, ?, ?, ?)', user)
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

def requires_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = dash.callback_context.inputs.get('session-token', None)
        if not token:
            return dash.no_update
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except:
            return dash.no_update
        return f(*args, **kwargs)
    return decorated_function

# Validate JWT token
def validate_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        return None

# Get user role from token
def get_user_role(token):
    decoded = validate_token(token)
    return decoded['role'] if decoded else None

# Get username from token
def get_username(token):
    decoded = validate_token(token)
    return decoded['user'] if decoded else None

# Login page layout
def get_login_layout():
    return html.Div([
        html.H1("HCT Academic Tracking System", className="text-center mt-4 mb-4"),
        dbc.Card([
            dbc.CardBody([
                dbc.Input(id='username-input', placeholder='Username', type='text', className='mb-3'),
                dbc.Input(id='password-input', placeholder='Password', type='password', className='mb-3'),
                dbc.Button('Login', id='login-button', color='primary', className='w-100'),
                html.Div(id='login-error', className='text-danger mt-3')
            ])
        ], className='mx-auto', style={'maxWidth': '400px'})
    ])

# Main app layout with auth
def get_auth_layout():
    return html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-token'),
        html.Div(id='page-content')
    ])

# Set up authentication callbacks
def setup_auth_callbacks(app):
    @app.callback(
        [Output('session-token', 'data'),
         Output('login-error', 'children'),
         Output('url', 'pathname')],
        [Input('login-button', 'n_clicks')],
        [State('username-input', 'value'),
         State('password-input', 'value')]
    )
    def login(n_clicks, username, password):
        if not n_clicks:
            raise PreventUpdate
        
        if not username or not password:
            return None, "Please enter both username and password", dash.no_update
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            c.execute('SELECT role FROM users WHERE username=? AND password=?', 
                    (username, hashed_password))
            result = c.fetchone()
            
            if result:
                c.execute('UPDATE users SET last_login=? WHERE username=?', 
                        (datetime.datetime.now(), username))
                conn.commit()
                
                token = jwt.encode(
                    {'user': username, 'role': result[0], 
                     'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=8)},
                    SECRET_KEY,
                    algorithm="HS256"
                )
                conn.close()
                return token, "", "/"
            
            conn.close()
            return None, "Invalid username or password", dash.no_update
        except Exception as e:
            print(f"Login error: {str(e)}")
            return None, "An error occurred during login", dash.no_update

    @app.callback(
        Output('session-token', 'clear_data'),
        [Input('logout-link', 'n_clicks')]
    )
    def logout(n_clicks):
        if n_clicks:
            return True
        raise PreventUpdate
        
