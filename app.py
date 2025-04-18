import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from flask import request, jsonify

# Optional pyodbc import
try:
    import pyodbc
except ImportError:
    print("pyodbc not installed. Using sample data only.")
    pyodbc = None

# Initialize the app
app = dash.Dash(__name__, title="Forecast Weight Dashboard", suppress_callback_exceptions=True)
server = app.server  # Expose Flask server to add custom routes

# Global variables to store the last received data
current_flight_data = {
    "flight_no": "",
    "flight_date": datetime.now().date().isoformat(),
    "flight_origin": "",
    "flight_destination": ""
}

# Layout - removed input fields and submit button, added display fields
app.layout = html.Div([
    dcc.Loading(id="loading-output", type="circle", children=[
        html.Div([
            html.H3("Forecasted Entry Point Condition - Weight", style={"textAlign": "center"}),
            dcc.Graph(id="weight-graph"),
        ])
    ]),
    
    # Hidden div to store the flight data from the .NET application
    html.Div(id="flight-data-store", style={"display": "none"}),
    
    # Add interval component to trigger updates
    dcc.Interval(
        id='interval-component',
        interval=180000,  # in milliseconds (3 minutes)
        n_intervals=0
    )
])

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
        
        # Try connecting to the database with a shorter timeout for faster fallback
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=qidtestingindia.database.windows.net;'
            'DATABASE=rm-demo-erp-db;'
            'UID=rmdemodeploymentuser;'
            'PWD=rm#demo#2515;'
            'Encrypt=yes;'
            'TrustServerCertificate=no;'
            'Connection Timeout=10;'  # Reduced timeout for faster response
        )
        
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
            SELECT ReportWeight
            FROM dbo.CapacityTransaction
            WHERE FltNo = ? AND CONVERT(date, FltDate) = ? AND Origin = ? AND Destination = ?
        """, (flight_no, formatted_date, origin, destination))
        
        capacity_record = cursor.fetchone()
        
        if capacity_record:
            capacity_weight = capacity_record[0]
            print(f"Found capacity record with ReportWeight: {capacity_weight}")
        else:
            print("No capacity record found. Will use estimated capacity.")
            # Use an estimate based on forecast data
            if not forecast_df.empty and 'EntryPointWeight' in forecast_df.columns:
                capacity_weight = forecast_df['EntryPointWeight'].mean() * 1.1  # 10% higher as an estimate
                print(f"Estimated capacity weight: {capacity_weight}")
        
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
    
    return forecast_df, capacity_weight

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
        
        print(f"Received data: {current_flight_data}")
        
        return jsonify({"status": "success", "message": "Data received successfully"}), 200
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# Callback to update the display of flight information
@callback(
    [Output("display-flight-number", "children"),
     Output("display-flight-date", "children"),
     Output("display-origin", "children"),
     Output("display-destination", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_flight_info(n_intervals):
    return (
        current_flight_data["flight_no"],
        current_flight_data["flight_date"],
        current_flight_data["flight_origin"],
        current_flight_data["flight_destination"]
    )

# Callback to update the graph based on stored flight data
@callback(
    Output("weight-graph", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_forecast_graph(n_intervals):
    # Get the current flight data
    flight_no = current_flight_data["flight_no"]
    flight_date = current_flight_data["flight_date"]
    origin = current_flight_data["flight_origin"]
    destination = current_flight_data["flight_destination"]
    
    if not all([flight_no, flight_date, origin, destination]):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": "Waiting for flight data...", "showarrow": False, "font": {"size": 16}}]
        )
        return empty_fig

    # Get both forecast data and capacity weight
    df, capacity_weight = get_forecast_and_capacity_data(flight_no, flight_date, origin, destination)

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": "No forecast data found for the given parameters.", "showarrow": False, "font": {"size": 16}}]
        )
        return empty_fig

    # Add error checking before using .dt accessor
    if pd.api.types.is_datetime64_any_dtype(df['ForecastDate']):
        df['Label'] = df['ForecastDate'].dt.strftime('%d %b').str.upper()
    else:
        # Handle the case where conversion to datetime failed
        print("Warning: ForecastDate column is not in datetime format. Attempting to convert...")
        df['ForecastDate'] = pd.to_datetime(df['ForecastDate'], errors='coerce')
        
        # Check if conversion was successful
        if pd.api.types.is_datetime64_any_dtype(df['ForecastDate']):
            df['Label'] = df['ForecastDate'].dt.strftime('%d %b').str.upper()
        else:
            # If conversion failed, use index as label
            print("Could not convert ForecastDate to datetime. Using index as label.")
            df['Label'] = [f"Day {i+1}" for i in range(len(df))]

    fig = go.Figure()
    
    # Add the forecast weight line
    fig.add_trace(go.Scatter(
        x=df['Label'], 
        y=df['EntryPointWeight'],
        mode='lines+markers', 
        name='Forecasted Weight',
        marker=dict(color='green'),
        line=dict(width=2)
    ))
    
    # Debug print to check capacity_weight value
    print(f"Capacity weight value: {capacity_weight}, type: {type(capacity_weight)}")
    
    # Force capacity_weight to be a number if it's None or not numeric
    if capacity_weight is None:
        # Default to 10% above max forecast weight if capacity is None
        capacity_weight = df['EntryPointWeight'].max() * 1.1
        print(f"Using default capacity weight: {capacity_weight}")
    
    try:
        # Try to convert to float to ensure it's numeric
        capacity_weight = float(capacity_weight)
        
        # Add the capacity weight as a horizontal line
        fig.add_trace(go.Scatter(
            x=df['Label'], 
            y=[capacity_weight] * len(df),
            mode='lines', 
            name='Predicted Capacity',
            line=dict(color='red', width=2, dash='dash')
        ))
        print(f"Successfully added capacity line with value: {capacity_weight}")
    except (ValueError, TypeError) as e:
        print(f"Error adding capacity line: {str(e)}")
    
    # Update layout with additional information
    fig.update_layout(
        xaxis_title="Forecast Date",
        yaxis_title="Weight (kg)",
        #title="Forecasted Weight vs. Predicted Capacity",
        xaxis=dict(
            tickmode='array',
            tickvals=df['Label'],
            ticktext=df['Label'],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0',port=int(os.environ.get('PORT',8050)))