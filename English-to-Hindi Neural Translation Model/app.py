from flask import Flask, render_template, request

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = TFAutoModelForSeq2SeqLM.from_pretrained(r"D:\\PYTHON\\PY eng 2 hin\\tf_model\\")


# Translation function
def translate_sentence(input_text):
    try:
        input_ids = tokenizer(input_text, return_tensors='tf').input_ids
        output_ids = model.generate(input_ids, max_length=128)
        translated_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return translated_sentence
    except Exception as e:
        return f"Error during translation: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html', input_text="", translated_result="")

@app.route('/translate', methods=['POST'])
def translate():
    try:
        input_text = request.form['input_text']
        translated_result = translate_sentence(input_text)
        return render_template('index.html', input_text=input_text, translated_result=translated_result)
    except Exception as e:
        error_message = f"Error during translation: {str(e)}"
        return render_template('index.html', input_text="", translated_result="", error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)

