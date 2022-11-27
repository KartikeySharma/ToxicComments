from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    sentence = [x for x in request.form.values()][0]
    t1= vectorizer.transform([sentence])
    prediction = model.predict(t1)

    output = prediction[0]
    
    if(output == 0):
        
        prediction_text = 'Toxicity label : 0 \n \n Comment Posted! \n \n Comment : {}'.format(sentence)
        return render_template('index.html', output=output,prediction_text = prediction_text.split('\n'))
    
    else:
        
        prediction_text='Toxicity label: 1 \n\n Comment Cannot be Posted!!! \n\n Please maintain decorum of the portal'
        return render_template('index.html', output=output,prediction_text = prediction_text.split('\n'))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)