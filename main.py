from google.cloud import datastore
import datetime
import random
from flask import Flask, render_template, redirect
import google.oauth2.id_token
from flask import Flask, render_template, request
from google.auth.transport import requests
from prediction import WeatherForecastModel

app = Flask(__name__)

#datastore_client = datastore.Client()
#firebase_request_adapter = requests.Request()
    
@app.route('/')
def root():

  return render_template('index.html')

@app.route('/weather_predict',methods=['POST'])
def weather_predict():
   #id_token = request.cookies.get("token")
   #error_message = None
   #claims = None
   #claims = google.oauth2.id_token.verify_firebase_token(id_token,
  #firebase_request_adapter)
   County = request.form['county']
   Station = request.form['station']
   Start = request.form['StartDate']
   End = request.form['EndDate']
   model = WeatherForecastModel('data/Book1.xlsx')
   forecast_df = model.train_and_forecast(County, Station, Start,End)
   forecast_html = forecast_df.to_html(classes='table table-striped', index=False)
   return render_template('final.html',forecast = forecast_html)

@app.route('/county/<county_name>')
def county(county_name):
    # Here you can add logic specific to each county
    if county_name == 'Dublin':
        return render_template ('dublin.html',county_name = county_name)
    elif county_name == 'Kerry':
        # Function for Dublin
        return render_template ('kerry.html',county_name = county_name)
    elif county_name == 'Galway':
        # Function for Cork
        return render_template ('galway.html',county_name = county_name)
    elif county_name == 'Mayo':
        # Function for Cork
        return render_template ('mayo.html',county_name = county_name)
    elif county_name == 'Cork':
        # Function for Cork
        return render_template ('cork.html',county_name = county_name)
    elif county_name == 'Clare':
        # Function for Cork
        return render_template ('clare.html',county_name = county_name)
    else:
        return "Unknown County"

@app.route('/station/<county_name>/<station_name>')
def station(county_name, station_name):
    county = county_name
    station = station_name
    # Perform actions with county_name and station_name
    return render_template('final.html',county = county,station = station)

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8080, debug=True)