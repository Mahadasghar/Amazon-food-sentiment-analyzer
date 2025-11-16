🍔 Amazon Food Review Sentiment Analyzer (Ensemble Model)

This project provides a robust, real-time sentiment analysis solution for food reviews, combining the speed of the VADER lexicon-based model with the high-context accuracy of a fine-tuned RoBERTa Transformer model into a single, high-performing Ensemble Classifier.

The system is deployed as a fully interactive Streamlit web application.

✨ Features

Ensemble Classification: Utilizes a weighted ensemble approach for superior accuracy and robustness compared to using either model in isolation.

Transformer Power: Incorporates a fine-tuned RoBERTa-base model on the Amazon Fine Food Reviews dataset for high-fidelity predictions.

Batch & Real-Time Analysis: Supports analysis for single-text inputs or bulk uploads via CSV files.

Interactive Dashboard: Features a Streamlit UI with dynamic charts (Pie Chart, Bar Chart) and detailed data tables.

Analysis History: Keeps a running history of all performed analyses within the session.

🚀 Quick Start (Local Deployment)

To run the Streamlit application locally, ensure you have Python 3.9+ installed.

1. Clone the Repository
```
git clone git@github.com:Mahadasghar/Amazon-food-sentiment-analyzer.git
cd Amazon-food-sentiment-analyzer
```

2. Install Dependencies

Install all required libraries using pip:
```
pip install -r requirements.txt.txt
```


3. Run the Application

Start the Streamlit application from your terminal:
```
streamlit run app.py
```

The application will automatically open in your web browser (usually at http://localhost:8501).

![Alt text for the image](https://github.com/Mahadasghar/Amazon-food-sentiment-analyzer/blob/main/interface.png)

💾 Model & Data

Model Weights

The core of this project is the fine-tuned RoBERTa model, which has been saved after training on the public Amazon Fine Food Reviews corpus.

Due to GitHub size limitations for file storage, the full model weights have been hosted externally for easy access by the app.py.

Drive access link 

[https://drive.google.com/file/d/1hzeMYdekYVtq8cROFPaOtzuEAhcg80ax/view?usp=drive_link]

Training Data: The model was trained on the Amazon Fine Food Reviews dataset.

Notebook: The end-to-end process of data loading, preprocessing, model fine-tuning (RoBERTa), VADER implementation, ensemble creation, and evaluation is documented in the following Jupyter Notebook: Amazon_food_review_sentiment_analysis_and_model_finetuning.ipynb

⚙️ Project Structure
```
.
├── Amazon_food_review_sentiment_analysis_and_model_finetuning.ipynb  
├── Test_for_app.csv                                                   
├── app.py                                                             
├── requirements.txt
├── Data_set.zip
├── Lisence                                               
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- 🎯 Improve model accuracy with advanced techniques
- 🌐 Add multi-language support
- 📊 Additional visualization features
- 🧪 More comprehensive test cases
- 📱 Mobile-responsive UI improvements

---

## 📝 Future Improvements

- [ ] Fine-tune on larger dataset (100k+ reviews)
- [ ] Deploy to cloud (Heroku/AWS/GCP)
- [ ] Add API endpoint for programmatic access
- [ ] Support for other product categories
- [ ] Real-time streaming analysis
- [ ] Multi-language sentiment detection

---

## 📚 References

### Dataset
- **Amazon Fine Food Reviews**: [Kaggle Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Citation**: J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.

### Models
- **RoBERTa**: [Liu et al., 2019 - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692)
- **VADER**: [Hutto & Gilbert, 2014 - "VADER: A Parsimonious Rule-based Model for Sentiment Analysis"](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf)



---



## 👤 Author

**Your Name**
- GitHub: (https://github.com/Mahadasghar))
- LinkedIn: (www.linkedin.com/in/muhammad-mahad-7b4b56260)
- Email: mahadasghar01@gmail.com

---

##  Acknowledgments

- Amazon for providing the food reviews dataset
- Hugging Face for the Transformers library
- Cardiff NLP for the pre-trained RoBERTa sentiment model
- Streamlit team for the amazing web framework
- The open-source community for invaluable tools and resources

---

## ⭐ Show Your Support

If you find this project helpful, please consider giving it a ⭐️!

---



<div align="center">
Made with ❤️ and 🤖 by [Muhammad Mahad ]
</div>
