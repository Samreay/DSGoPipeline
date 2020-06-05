import dash_html_components as html
import dash_core_components as dcc

from components.elements import get_card, get_rolling_plot, get_predictions


def get_location_base(app):
    card1 = get_card("col-sm-12", component=dcc.Markdown("""
# Power Prediction Dashboard

Some super simple Dash app here. Im just writing
this section out in *markdown* because I hate the way
you're supposed to define HTML components in python. 

If you also dislike it, its easy to rip out Dash and have this
as a base flask app, which will allow you to create some get 
endpoints which will be more efficiently cached.
"""))
    card2 = get_rolling_plot(app, "card_rolling", class_name="col-xl-6")
    card3 = get_predictions(app, "card_predictions", class_name="col-xl-6")

    return [html.Div(className="row", children=[card1, card2, card3])]
