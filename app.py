import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from flask import request, jsonify
import os
import traceback
import decimal

# Optional pyodbc import
try:
    import pyodbc
except ImportError:
    print("pyodbc not installed. Using sample data only.")
    pyodbc = None

# Initialize the app
app = dash.Dash(__name__, 
                title="Forecast Weight Dashboard", 
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose Flask server to add custom routes

# Global variables to store the last received data
current_flight_data = {
    "flight_no": "",
    "flight_date": datetime.now().date().isoformat(),
    "flight_origin": "",
    "flight_destination": ""
}

# Layout - full viewport with loading circle
app.layout = html.Div([
    # Store component to store the flight data
    dcc.Store(id='flight-data-store'),
    
    # Graph container with responsive layout and loading overlay
    dcc.Loading(
        id="loading-graph",
        type="circle",
        color="#119DFF",
        children=[
            html.Div(
                id="graph-container",
                style={
                    "width": "100%", 
                    "height": "100vh",  # Use viewport height
                    "padding": "0px",   # Remove padding
                    "margin": "0px"     # Remove margin
                }
            )
        ]
    ),
    
    # Add interval component to trigger updates
    dcc.Interval(
        id='interval-component',
        interval=1200000,  # in milliseconds (20 minutes)
        n_intervals=0
    ),
    
    # Location component to track URL
    dcc.Location(id='url', refresh=False),
    
    # Hidden div to trigger callback on page load
    html.Div(id='page-load-trigger', style={'display': 'none'})
], style={
    "width": "100%",
    "height": "100vh",  # Use full viewport height
    "padding": "0px",   # Remove padding
    "margin": "0px",    # Remove margin
    "overflow": "hidden" # Prevent scrollbars
})

# Initialize Flask routes BEFORE Dash callbacks
@server.before_request
def before_request_func():
    # Process query parameters on every request
    query_params = request.args
    if query_params:
        # Debug log
        print(f"INTERCEPTED QUERY PARAMS: {query_params}")
        
        # Update the global variable
        global current_flight_data
        
        # Check if we're resetting the dashboard
        if query_params.get('reset') == 'true':
            print("Resetting dashboard data via query parameter")
            current_flight_data = {
                "flight_no": "",
                "flight_date": datetime.now().date().isoformat(),
                "flight_origin": "",
                "flight_destination": ""
            }
        # Check if we have flight parameters
        elif all(param in query_params for param in ['flight_no', 'flight_date', 'flight_origin', 'flight_destination']):
            print("Updating dashboard data via query parameters")
            current_flight_data = {
                "flight_no": query_params.get('flight_no', ''),
                "flight_date": query_params.get('flight_date', datetime.now().date().isoformat()),
                "flight_origin": query_params.get('flight_origin', ''),
                "flight_destination": query_params.get('flight_destination', '')
            }
            print(f"Updated flight data to: {current_flight_data}")

# Callback to update the data store from URL
@app.callback(
    Output('flight-data-store', 'data'),
    [Input('url', 'pathname'),
     Input('url', 'search'),
     Input('interval-component', 'n_intervals')]
)
def update_store_from_url(pathname, search, n_intervals):
    # Return the current flight data from the global variable
    return current_flight_data

def get_forecast_and_capacity_data(flight_no, flight_date, origin, destination):
    """
    Retrieve forecast data from database and capacity data, or generate sample data if DB connection fails.
    
    Returns:
    - forecast_df: DataFrame with forecast data
    - capacity_weight: Static value representing the Predicted capacity (ReportWeight)
    """
    forecast_df = None
    capacity_weight = None
    
    try:
        # Check if pyodbc is available
        if pyodbc is None:
            print("pyodbc module not available. Using sample data.")
            raise ImportError("pyodbc is not available")
            
        print("Attempting to connect to database...")
        
        # Get database connection details from environment variables
        db_server = os.environ.get('DB_SERVER', '')
        db_name = os.environ.get('DB_NAME', '')
        db_user = os.environ.get('DB_USER', '')
        db_password = os.environ.get('DB_PASSWORD', '')
        
        # Check if we have all the required connection details
        if not all([db_server, db_name, db_user, db_password]):
            print("Missing database connection details. Using sample data instead...")
            raise Exception("Missing database connection details")
        
        # Try connecting to the database
        conn_str = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={db_server};'
            f'DATABASE={db_name};'
            f'UID={db_user};'
            f'PWD={db_password};'
            f'Encrypt=yes;'
            f'TrustServerCertificate=no;'
            f'Connection Timeout=30;'
        )
        
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Convert flight_date to datetime if it's a string
        if isinstance(flight_date, str):
            try:
                flight_date = datetime.strptime(flight_date.split('T')[0], "%Y-%m-%d").date()
            except:
                flight_date = datetime.now().date()
                
        formatted_date = flight_date.strftime("%Y-%m-%d")
        
        print(f"Connected to database. Querying for flight {flight_no} on {formatted_date}...")
        
        # Get forecast data
        cursor.execute("""
            SELECT FltNumber, FltDate, FltOrigin, FltDestination, ForecastDate,
                   EntryPointWeight, EntryPointVolume
            FROM dbo.AirlineEntryPointCondition
            WHERE FltNumber = ? AND FltDate = ? AND FltOrigin = ? AND FltDestination = ?
            ORDER BY ForecastDate
        """, (flight_no, formatted_date, origin, destination))

        forecast_records = cursor.fetchall()
        
        if not forecast_records:
            print("No forecast records found in database. Using sample data.")
            raise ValueError("No forecast data found in database query")
            
        forecast_columns = [col[0] for col in cursor.description]
        forecast_df = pd.DataFrame.from_records(forecast_records, columns=forecast_columns)
        
        # Explicitly convert ForecastDate to datetime
        forecast_df['ForecastDate'] = pd.to_datetime(forecast_df['ForecastDate'], errors='coerce')
        
        # Get capacity data - using CONVERT to handle date with time
        cursor.execute("""
            SELECT ReportWeight, OBW
            FROM dbo.CapacityTransaction
            WHERE FltNo = ? AND CONVERT(date, FltDate) = ? AND Origin = ? AND Destination = ?
        """, (flight_no, formatted_date, origin, destination))
        
        capacity_record = cursor.fetchone()
        
        if capacity_record:
            capacity_weight = capacity_record[0]
            obw_weight = capacity_record[1] if len(capacity_record) > 1 else None
            print(f"Found capacity record with ReportWeight: {capacity_weight}, OBW: {obw_weight}")
        else:
            print("No capacity record found. Will use estimated capacity.")
            # Use an estimate based on forecast data
            if not forecast_df.empty and 'EntryPointWeight' in forecast_df.columns:
                capacity_weight = forecast_df['EntryPointWeight'].mean() * 1.1  # 10% higher as an estimate
                obw_weight = capacity_weight * 0.95  # Slightly lower than capacity as an estimate
                print(f"Estimated capacity weight: {capacity_weight}, estimated OBW: {obw_weight}")
            else:
                capacity_weight = None
                obw_weight = None
        
        cursor.close()
        conn.close()
        
        print(f"Successfully retrieved {len(forecast_df)} forecast records from database.")
        
    except Exception as e:
        print(f"Database error: {str(e)}\nGenerating sample data instead...")
        
        # Create sample data based on input parameters
        today = datetime.now().date()
        forecast_dates = [today + timedelta(days=i) for i in range(10)]
        
        # Generate weights with a realistic pattern (increasing then decreasing)
        base_weight = 750 + (ord(flight_no[0]) % 10) * 25 if flight_no else 800
        weights = []
        for i in range(10):
            # Create a curve that peaks in the middle
            deviation = 100 * (1 - abs(i - 4.5) / 4.5)
            weights.append(int(base_weight + deviation))
        
        # Create sample DataFrame with proper datetime objects
        forecast_df = pd.DataFrame({
            "FltNumber": [flight_no] * 10,
            "FltDate": [flight_date] * 10,
            "FltOrigin": [origin] * 10,
            "FltDestination": [destination] * 10,
            "ForecastDate": pd.to_datetime(forecast_dates),
            "EntryPointWeight": weights,
        })
        
        # Generate a sample capacity weight (slightly higher than the max forecast weight)
        capacity_weight = max(weights) * 1.1
        
        print(f"Generated sample data with {len(forecast_df)} records.")
        print(f"Generated sample capacity weight: {capacity_weight}")
    
    return forecast_df, capacity_weight, obw_weight

def get_actual_bookings_data(flight_no, flight_date, origin, destination):
    """
    Retrieve actual booking data from database, or generate sample data if DB connection fails.
    
    Returns:
    - bookings_df: DataFrame with booking dates and aggregated weights
    """
    try:
        # Check if pyodbc is available
        if pyodbc is None:
            print("pyodbc module not available. Using sample booking data.")
            raise ImportError("pyodbc is not available")
        
        print("Attempting to connect to database for actual bookings data...")
        
        # Get database connection details from environment variables
        db_server = os.environ.get('DB_SERVER', '')
        db_name = os.environ.get('DB_NAME', '')
        db_user = os.environ.get('DB_USER', '')
        db_password = os.environ.get('DB_PASSWORD', '')
        
        
        # Check if we have all the required connection details
        if not all([db_server, db_name, db_user, db_password]):
            print("Missing database connection details. Using sample booking data instead...")
            raise Exception("Missing database connection details")
        
        # Try connecting to the database
        conn_str = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={db_server};'
            f'DATABASE={db_name};'
            f'UID={db_user};'
            f'PWD={db_password};'
            f'Encrypt=yes;'
            f'TrustServerCertificate=no;'
            f'Connection Timeout=30;'
        )
        
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Convert flight_date to datetime if it's a string
        if isinstance(flight_date, str):
            try:
                flight_date_obj = datetime.strptime(flight_date.split('T')[0], "%Y-%m-%d").date()
            except:
                flight_date_obj = datetime.now().date()
        else:
            flight_date_obj = flight_date
                
        formatted_date = flight_date_obj.strftime("%Y-%m-%d")
        next_day = (flight_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"Connected to database. Querying for actual bookings for flight {flight_no} from {origin} to {destination} on {formatted_date}...")
        
        # Execute the query for actual bookings data
        cursor.execute("""
        SELECT ARM.[AWBID],
        ARM.[AWBPrefix],
        ARM.[AWBNumber],
        ARM.[FltNumber],
        ARM.[FltDate],
        ARM.[FltOrigin],
        ARM.[FltDestination],
        ARM.[Wt],
        ARM.[UOM],
        ARM.[allotmentcode],
        ARM.[Volume],
        ARM.[VolumeUnit],
        AT.[AWBDate]
        FROM dbo.AWBRouteMaster ARM 
        LEFT JOIN dbo.AWBSummaryMaster AT ON ARM.AWBPrefix = AT.AWBPrefix AND ARM.AWBNumber = AT.AWBNumber 
        WHERE ARM.FltOrigin = ? AND ARM.FltDestination = ? AND ARM.FltNumber = ? 
        AND ARM.FltDate >= ? AND ARM.FltDate <= ?
        """, (origin, destination, flight_no, formatted_date, next_day))
        
        booking_records = cursor.fetchall()
        
        if not booking_records:
            print("No booking records found in database. Will use sample booking data.")
            raise ValueError("No booking data found in database query")
            
        booking_columns = [col[0] for col in cursor.description]
        raw_bookings_df = pd.DataFrame.from_records(booking_records, columns=booking_columns)
        
        print(f"Successfully retrieved {len(raw_bookings_df)} booking records from database.")
        
        # Convert AWBDate to datetime
        raw_bookings_df['AWBDate'] = pd.to_datetime(raw_bookings_df['AWBDate'], errors='coerce')
        
        # Extract day and month to match the forecast data format
        raw_bookings_df['BookingDay'] = raw_bookings_df['AWBDate'].dt.strftime('%d %b').str.upper()
        
        # Also store the full datetime for sorting
        raw_bookings_df['BookingDateFull'] = raw_bookings_df['AWBDate']
        
        # Group by the day and sum weights
        bookings_df = raw_bookings_df.groupby('BookingDay', as_index=False)['Wt'].sum()
        bookings_df.rename(columns={'Wt': 'DailyBookedWeight'}, inplace=True)
        
        # Add the full date
        bookings_df['BookingDateFull'] = pd.to_datetime(raw_bookings_df.groupby('BookingDay')['BookingDateFull'].first().values)
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Database error for bookings: {str(e)}\nGenerating sample booking data instead...")
        
        # Create sample booking data
        # Generate dates around the flight date
        base_date = datetime.now().date() if isinstance(flight_date, str) else flight_date
        if isinstance(base_date, str):
            try:
                base_date = datetime.strptime(base_date.split('T')[0], "%Y-%m-%d").date()
            except:
                base_date = datetime.now().date()
        
        # Generate dates 5 days before and 5 days after
        booking_dates = [base_date - timedelta(days=5-i) for i in range(10)]
        booking_labels = [d.strftime('%d %b').upper() for d in booking_dates]
        
        # Generate booking weights that gradually increase (simulating more bookings as flight date approaches)
        base_weight = 100 + (ord(flight_no[0]) % 10) * 5 if flight_no else 120
        booking_weights = []
        for i in range(10):
            # Create an increasing curve for daily bookings
            weight = base_weight * (0.6 + 0.05 * i)  # Starts at 60% and increases
            booking_weights.append(int(weight))
        
        # Create sample bookings DataFrame
        bookings_df = pd.DataFrame({
            'BookingDay': booking_labels,
            'DailyBookedWeight': booking_weights,
            'BookingDateFull': booking_dates
        })
        
        print(f"Generated sample booking data with {len(bookings_df)} records.")
    
    return bookings_df

# KEEP THE ORIGINAL POST ENDPOINTS FOR BACKWARD COMPATIBILITY

# API endpoint to receive data from .NET application
@server.route('/update-data', methods=['POST'])
def update_data():
    global current_flight_data
    
    try:
        # Get the data from the request
        data = request.get_json()
        
        # Update the current flight data
        current_flight_data = {
            "flight_no": data.get("flight_no", ""),
            "flight_date": data.get("flight_date", datetime.now().date().isoformat()),
            "flight_origin": data.get("flight_origin", ""),
            "flight_destination": data.get("flight_destination", "")
        }
        
        print(f"Received data via POST: {current_flight_data}")
        
        return jsonify({"status": "success", "message": "Data received successfully"}), 200
    
    except Exception as e:
        print(f"Error processing POST data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# API endpoint to reset data
@server.route('/reset-data', methods=['POST'])
def reset_data():
    global current_flight_data
    
    try:
        # Reset the current flight data to empty values
        current_flight_data = {
            "flight_no": "",
            "flight_date": datetime.now().date().isoformat(),
            "flight_origin": "",
            "flight_destination": ""
        }
        
        print("Dashboard data reset via POST")
        
        return jsonify({"status": "success", "message": "Data reset successfully"}), 200
    
    except Exception as e:
        print(f"Error resetting data via POST: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# Callback to update the graph based on stored flight data
@app.callback(
    Output("graph-container", "children"),
    [Input("flight-data-store", "data")]
)
def update_forecast_graph(flight_data):
    try:
        # Use the flight data from the store if available
        if flight_data:
            flight_no = flight_data.get("flight_no", "")
            flight_date = flight_data.get("flight_date", "")
            origin = flight_data.get("flight_origin", "")
            destination = flight_data.get("flight_destination", "")
        else:
            # Fall back to global variable if store is empty
            flight_no = current_flight_data["flight_no"]
            flight_date = current_flight_data["flight_date"]
            origin = current_flight_data["flight_origin"]
            destination = current_flight_data["flight_destination"]
        
        # For debugging
        print(f"Updating forecast graph with: Flight={flight_no}, Date={flight_date}, Origin={origin}, Dest={destination}")
        
        if not all([flight_no, flight_date, origin, destination]):
            return html.Div("Waiting for flight data...", 
                            style={
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "center",
                                "height": "100%",
                                "fontSize": "16px"
                            })

        # Get forecast data and capacity weight
        df, capacity_weight, obw_weight = get_forecast_and_capacity_data(flight_no, flight_date, origin, destination)
        
        # Get actual bookings data
        bookings_df = get_actual_bookings_data(flight_no, flight_date, origin, destination)

        if df.empty:
            return html.Div(f"No forecast data found for flight {flight_no} from {origin} to {destination} on {flight_date}.", 
                            style={
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "center",
                                "height": "100%",
                                "fontSize": "16px"
                            })

        # Add error checking before using .dt accessor
        if pd.api.types.is_datetime64_any_dtype(df['ForecastDate']):
            df['Label'] = df['ForecastDate'].dt.strftime('%d %b').str.upper()
            df['DateSort'] = df['ForecastDate']  # For sorting
        else:
            # Handle the case where conversion to datetime failed
            print("Warning: ForecastDate column is not in datetime format. Attempting to convert...")
            df['ForecastDate'] = pd.to_datetime(df['ForecastDate'], errors='coerce')
            
            # Check if conversion was successful
            if pd.api.types.is_datetime64_any_dtype(df['ForecastDate']):
                df['Label'] = df['ForecastDate'].dt.strftime('%d %b').str.upper()
                df['DateSort'] = df['ForecastDate']  # For sorting
            else:
                # If conversion failed, use index as label
                print("Could not convert ForecastDate to datetime. Using index as label.")
                df['Label'] = [f"Day {i+1}" for i in range(len(df))]
                df['DateSort'] = pd.Series(range(len(df)))  # For sorting

        # Sort the forecast dataframe by date
        df.sort_values('DateSort', inplace=True)

        # Calculate total booked weight for all dates before the first forecast date
        # First, we need a date mapping from labels to full dates
        if 'BookingDateFull' not in bookings_df.columns:
            # If sample data doesn't have full dates, create them from the labels
            try:
                current_year = datetime.now().year
                bookings_df['BookingDateFull'] = bookings_df['BookingDay'].apply(
                    lambda x: datetime.strptime(f"{x} {current_year}", '%d %b %Y')
                )
            except Exception as e:
                print(f"Error creating full dates from labels: {e}")
                # Just use sequential numbers if date conversion fails
                bookings_df['BookingDateFull'] = pd.Series(range(len(bookings_df)))

        # Sort bookings by date
        bookings_df.sort_values('BookingDateFull', inplace=True)
        
        # Create a mapping from date labels to sequences for proper sorting
        date_order = {label: i for i, label in enumerate(df['Label'])}
        
        # Merge with forecast dates to ensure we have all the dates from the forecast
        merged_df = pd.merge(
            df[['Label', 'DateSort']], 
            bookings_df, 
            how='left', 
            left_on='Label', 
            right_on='BookingDay'
        )
        
        # Fill NaN values with 0 for dates where there were no bookings
        merged_df['DailyBookedWeight'].fillna(0, inplace=True)
        
        # Sort by date for proper cumulative sum
        merged_df.sort_values('DateSort', inplace=True)
        
        # Calculate the cumulative sum of bookings
        merged_df['CumulativeBookedWeight'] = merged_df['DailyBookedWeight'].cumsum()
        
        # Include previous bookings (before first forecast date) in the cumulative total
        # This simulates having some bookings already in place before the forecast window
        previous_bookings = 200  # Example value - adjust as needed
        merged_df['CumulativeBookedWeight'] = merged_df['CumulativeBookedWeight'] + previous_bookings

        # Get current date for "Today" marker
        today_date = datetime.now()
        today_label = today_date.strftime('%d %b').upper()
        
        # Find closest date in the forecast data to today
        # This is needed because the forecast data might not have an exact match for today
        today_idx = None
        today_found = False
        
        # First check if today is in the labels
        if today_label in merged_df['Label'].values:
            today_idx = merged_df[merged_df['Label'] == today_label].index[0]
            today_found = True
        
        # If today is not found in labels, find the closest date
        if not today_found:
            # Convert labels to datetime for comparison
            try:
                # Try to convert the labels to datetime
                current_year = datetime.now().year
                label_dates = {}
                for label in merged_df['Label']:
                    try:
                        label_date = datetime.strptime(f"{label} {current_year}", '%d %b %Y')
                        label_dates[label] = label_date
                    except:
                        print(f"Could not parse date from label: {label}")
                
                # Find the closest date to today
                if label_dates:
                    closest_label = min(label_dates.keys(), key=lambda x: abs(label_dates[x] - today_date))
                    today_idx = merged_df[merged_df['Label'] == closest_label].index[0]
                    today_found = True
                    today_label = closest_label  # Use the closest label as today's label
            except Exception as e:
                print(f"Error finding closest date to today: {e}")
        
        # Truncate the actual bookings data to only show up to today
        if today_found:
            # Get the index of today in the merged dataframe
            today_merged_idx = merged_df.index.get_loc(today_idx)
            
            # Create a version of the merged dataframe with data only up to today
            merged_df_today = merged_df.iloc[:today_merged_idx + 1]
        else:
            # If we couldn't find today, use a fallback approach
            # For sample data, assume half of the dates are in the past
            half_idx = len(merged_df) // 2
            merged_df_today = merged_df.iloc[:half_idx]
            # Use the last date in this truncated dataframe as "today"
            today_label = merged_df_today['Label'].iloc[-1]

        fig = go.Figure()
        
        # Add the forecast weight line
        fig.add_trace(go.Scatter(
            x=df['Label'], 
            y=df['EntryPointWeight'],
            mode='lines+markers', 
            name='Forecast',
            marker=dict(color='green'),
            line=dict(width=2,color='green'),
            hovertemplate = 'Date: %{x}<br>Weight: %{y} Kg<extra></extra>'
        ))
        
        # Force capacity_weight to be a number if it's None or not numeric
        if capacity_weight is None:
            # Default to 10% above max forecast weight if capacity is None
            capacity_weight = float(df['EntryPointWeight'].max()) * 1.1
            print(f"Using default capacity weight: {capacity_weight}")
        else:
            # Convert to float if it's a Decimal
            if isinstance(capacity_weight, decimal.Decimal):
                capacity_weight = float(capacity_weight)
        
        try:
            # Try to convert to float to ensure it's numeric
            capacity_weight = float(capacity_weight)
            
            # Add the capacity weight as a horizontal line
            fig.add_trace(go.Scatter(
                x=df['Label'], 
                y=[capacity_weight] * len(df),
                mode='lines', 
                name='Pred Capacity',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate= 'Pred Capacity: %{y} Kg<extra></extra>'
            ))
        except (ValueError, TypeError) as e:
            print(f"Error adding capacity line: {str(e)}")
        
        if obw_weight is not None:
            try:
                # Convert to float if it's a Decimal
                if isinstance(obw_weight, decimal.Decimal):
                    obw_weight = float(obw_weight)
                    
                # Add the OBW line
                fig.add_trace(go.Scatter(
                    x=df['Label'], 
                    y=[obw_weight] * len(df),
                    mode='lines', 
                    name='OBW',
                    line=dict(color='orange', width=2, dash='dot'),
                    hovertemplate= 'OBW: %{y} Kg<extra></extra>'
                ))
            except (ValueError, TypeError) as e:
                print(f"Error adding OBW line: {str(e)}")
            
        # Plot cumulative bookings line if we have valid booking data (only up to today)
        if not merged_df_today.empty and 'CumulativeBookedWeight' in merged_df_today.columns:
            # Add the cumulative bookings line (only up to today)
            fig.add_trace(go.Scatter(
                x=merged_df_today['Label'], 
                y=merged_df_today['CumulativeBookedWeight'],
                mode='lines+markers', 
                name='Actual Bookings (Cumulative)',
                marker=dict(color='blue'),
                line=dict(width=2, color='blue'),
                hovertemplate = 'Date: %{x}<br>Total Booked Weight: %{y} Kg<extra></extra>'
            ))
            
            # Get the last known actual booking weight to use for the "Today" annotation
            latest_actual_weight = merged_df_today['CumulativeBookedWeight'].iloc[-1]
            
            # Add a "Today" annotation to the last point on the actual bookings line
            fig.add_annotation(
                x=today_label,
                y=latest_actual_weight,
                text="Today",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=0,
                ay=-40,
                font=dict(
                    family="Arial",
                    size=12,
                    color="black"
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="center"
            )
            
            # Add a vertical line at today's date
            fig.add_shape(
                type="line",
                x0=today_label,
                y0=0,
                x1=today_label,
                y1=max(float(capacity_weight), float(df['EntryPointWeight'].max()), float(latest_actual_weight)) * 1.1,
                line=dict(
                    color="black",
                    width=1,
                    dash="dot",
                ),
            )
        
        # Update layout to match other dashboards
        fig.update_layout(
            title=dict(
                text=f'Forecasted Entry Point Condition - Weight: {flight_no} {origin}-{destination}',
                x=0.5,  # Center title
                y=0.98  # Position near top
            ),
            xaxis_title="Forecast Date",
            yaxis_title="Weight (kg)",
            xaxis=dict(
                tickmode='array',
                tickvals=df['Label'],
                ticktext=df['Label'],
                tickangle=45,
            ),
            legend=dict(
                x=1.05,        # Just outside the right side
                y=1,           # Align to top
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background
                bordercolor='black',
                borderwidth=1
            ),
            template='plotly_white',
            margin=dict(l=50, r=100, t=60, b=50),
            autosize=True,
            height=None
        )
        
        return dcc.Graph(
            figure=fig,
            style={
                'height': '100%',  # Take full height of parent container
                'width': '100%'    # Take full width of parent container
            },
            config={
                'responsive': True,  # Enable responsiveness
                'displayModeBar': False  # Hide the mode bar for cleaner appearance
            }
        )
    
    except Exception as e:
        # Get full stack trace for debugging
        stack_trace = traceback.format_exc()
        print(f"Error: {e}")
        print(f"Stack trace: {stack_trace}")
        
        return html.Div(f"Error updating graph: {str(e)}", 
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "height": "100%",
                            "fontSize": "16px",
                            "color": "red"
                        })

# Add route to handle root path and query parameters
@server.route('/')
def index():
    # This is just to log when the root route is accessed with query parameters
    if request.args:
        print(f"Root route accessed with query parameters: {request.args}")
    return app.index()

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))