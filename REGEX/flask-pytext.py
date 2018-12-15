import sys
import flask
import pytext

config_file = sys.argv[1]
model_file = sys.argv[2]

config = pytext.load_config(config_file)
predictor = pytext.create_predictor(config, model_file)

app = flask.Flask(__name__)

@app.route('/get_flight_info', methods=['GET', 'POST'])
def get_flight_info():
    text = flask.request.data.decode()

    # Pass the inputs to PyText's prediction API
    result = predictor({"raw_text": text})

    # Results is a list of output blob names and their scores.
    # The blob names are different for joint models vs doc models
    # Since this tutorial is for both, let's check which one we should look at.
    doc_label_scores_prefix = (
        'scores:' if any(r.startswith('scores:') for r in result)
        else 'doc_scores:'
    )

    # For now let's just output the top document label!
    best_doc_label = max(
        (label for label in result if label.startswith(doc_label_scores_prefix)),
        key=lambda label: result[label][0],
    # Strip the doc label prefix here
    )[len(doc_label_scores_prefix):]

    return flask.jsonify({"question": f"Are you asking about {best_doc_label}?"})

app.run(host='0.0.0.0', port='8080', debug=True)
