import dash_html_components as html
import dash_core_components as dcc
import plotly
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from functools import lru_cache
from datetime import datetime as dt

import plotly.graph_objs as go
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def get_sideitem(name, url):
    return html.Li(className="nav-item", children=dcc.Link(className="nav-link", href=f"{url}", children=html.Span(children=name)))


def get_card(className, component=None):
    return html.Div(className=className, children=[html.Div(className="card shadow mb-4", children=[html.Div(className="card-body", children=component)])])


@lru_cache(maxsize=1)
def get_data():
    print("Loading data")
    df = pd.read_csv("../germany.csv", parse_dates=[0], index_col=0)
    return df


@lru_cache(maxsize=1)
def get_model():
    historical = get_data()
    X = historical[["windspeed", "temperature", "rad_horizontal", "rad_diffuse"]]
    y = historical[["solar_GW", "wind_GW"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(x):
    model = get_model()
    return pd.DataFrame(model.predict(x), columns=["solar_GW", "wind_GW"], index=x.index)


def get_rolling_plot(app, id, class_name="col-sm-12"):
    def get_fig():
        historical = get_data()

        fig = go.Figure()
        for i in range(2013, 2017):
            name = str(i)
            df = historical[name]
            df.index = df.index.map(lambda t: t.replace(year=2016))
            rolling = df.solar_GW.rolling(24 * 7).mean()

            fig.add_trace(go.Scatter(x=rolling.index, y=rolling, mode="lines", name=name))

        fig.update_layout(title="Solar power production", xaxis_title="Month",
                          yaxis_title="GW", margin=dict(l=0, r=0, t=30, b=0), height=600)
        fig.update_xaxes(rangeslider_visible=True, tickformatstops=[
            dict(dtickrange=[None, 604800000], value="%d-%B"),
            dict(dtickrange=[604800000, None], value="%B"),
        ])
        return fig

    @app.callback(Output(id, "figure"), [Input("refresh-interval", "n_intervals")])
    def update_plot(n):
        return get_fig()

    return get_card(class_name, component=dcc.Graph(id=id, figure=get_fig(), config={"displayModeBar": False}))


def get_predictions(app, id, class_name="col-sm-12"):
    picker = dcc.DatePickerRange(
        id='date_range_widget',
        min_date_allowed=dt(2013, 1, 1),
        max_date_allowed=dt(2017, 12, 31),
        start_date=dt(2016, 1, 1).date(),
        end_date=dt(2016, 2, 1).date()
    )

    def get_fig(start_date, end_date):
        historical = get_data()
        to_predict = historical[start_date:end_date]

        predicted = predict(to_predict[["windspeed", "temperature", "rad_horizontal", "rad_diffuse"]])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Solar Power", "Wind Power"))

        fig.add_trace(go.Scatter(x=to_predict.index, y=to_predict["solar_GW"], name="Recorded", line=dict(color="red")), row=1, col=1)
        fig.add_trace(go.Scatter(x=predicted.index, y=predicted["solar_GW"], name="Model", line=dict(color="blue")), row=1, col=1)

        fig.add_trace(go.Scatter(x=to_predict.index, y=to_predict["wind_GW"], name="Recorded", line=dict(color="red"), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=predicted.index, y=predicted["wind_GW"], name="Model", line=dict(color="blue"), showlegend=False), row=2, col=1)

        fig.update_layout(yaxis_title="GW", yaxis2_title="GW", margin=dict(l=0, r=0, t=30, b=0), height=550)
        return fig

    @app.callback(Output(id, "figure"),
                  [Input("refresh-interval", "n_intervals"), Input("date_range_widget", "start_date"), Input("date_range_widget", "end_date")])
    def update_plot(n, start_date, end_date):
        return get_fig(start_date, end_date)

    components = ["Pick your date range: ", picker, dcc.Graph(id=id, config={"displayModeBar": False})]
    return get_card(class_name, component=components)
