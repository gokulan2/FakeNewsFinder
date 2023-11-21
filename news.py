from flask import Flask, render_template, request, send_from_directory
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer

import newspaper

app = Flask(__name__)

app.static_folder = 'templates'

@app.route('/videos/<path:filename>')
def ytgiphy(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/news', methods=['GET','POST'])
def detect_fake_news():
    user_input = ""
    
    if request.method == 'POST':
        
        # Load the pre-trained BERT model and tokenizer for fake news detection
        bert_model_name = "bert-base-uncased"
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Load a pre-trained T5 model and tokenizer for generating explanations
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # Define a text classification pipeline
        classifier = pipeline('sentiment-analysis', model=bert_model, tokenizer=bert_tokenizer)

        # Function to analyze fake news
        def analyze_fake_news(news_text):
            try:
                # Classify the news text
                result = classifier(news_text)
                return result
            except Exception as e:
                return str(e)

        # Function to scrape a news article from a URL
        def scrape_article(url):
            article = newspaper.Article(url)
            article.download()
            article.parse()
            return article.text

        # Function to generate explanations using T5
        def generate_explanation(news_text, classification_result):
            input_text = f"Explain why the news article is classified as {classification_result}: {news_text}"

            # Tokenize and generate an explanation
            input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=200, truncation=True)
            generated_ids = t5_model.generate(input_ids, max_length=200, num_return_sequences=1, early_stopping=False)

            explanation = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return explanation

        try:
            user_input = request.form['article_url']
            analysis_result = analyze_fake_news(user_input)

            if isinstance(analysis_result, str):
                return render_template('news.html', response=f"Error analyzing fake news: {analysis_result}")
            else:
                label = analysis_result[0]['label']
                confidence = analysis_result[0]['score'] * 100
                confidence_threshold = 70  # Set your desired threshold here
                is_fake = confidence >= confidence_threshold

                labels = ['Real', 'Fake']

                predicted_label = 1 if is_fake else 0

                # Scraping the news article from a URL (you can replace the URL)
                article_url = user_input
                news_text = scrape_article(article_url)

                # Generating an explanation for the classification result
                explanation = generate_explanation(news_text, label)

                return render_template('news.html', news_text=user_input, label=label, confidence=confidence, predicted_label=predicted_label, is_fake=is_fake, explanation=explanation, labels=labels)

        except Exception as e:
            print(f"Error: {str(e)}")

    return render_template('news.html')

if __name__ == '__main__':
    app.run(debug=True)
