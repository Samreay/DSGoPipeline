import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from components.pages import get_location_base
from components.elements import get_sideitem
import os

app = dash.Dash("DSGo Dashboard")
app.title = "DSGo Dashboard"
app.layout = html.Div(
    id="wrapper",
    children=[
        dcc.Location(id="url", refresh=False),
        html.Ul(
            className="navbar-nav bg-gradient-primary sidebar sidebar-dark accordion",
            children=[
                html.Li(
                    children=html.A(
                        className="sidebar-brand d-flex align-items-center justify-content-center",
                        href="#",
                        children=html.Div(className="sidebar-brand-text mx-3", children="DSGo Dashboard"),
                    )
                ),
                get_sideitem("Landing Page", ""),
            ],
        ),
        html.Div(
            id="content-wrapper",
            className="d-flex flex-column",
            children=[
                html.Div(
                    id="content",
                    className="mt-4",
                    children=html.Div(className="container-fluid", id="fluidContainer"),
                )
            ],
        ),
        dcc.Interval(id="refresh-interval", interval=3600 * 1000, n_intervals=0),
    ],
)

layouts = {
    "": get_location_base(app)
}


@app.callback(Output("fluidContainer", "children"), [Input("url", "pathname")])
def display_dashboard(pathname):
    if pathname is None or pathname == "/":
        pathname = ""
    return layouts[pathname]


if __name__ == "__main__":
    app.config["suppress_callback_exceptions"] = True
    app.run_server(debug=True)
