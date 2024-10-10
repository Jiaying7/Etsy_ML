# Multimodal Embedding Extraction for Product Attribute Prediction on Etsy

This project implements machine learning techniques to predict product attributes, such as top category, primary color, and secondary color, based on a subset of Etsy's product listings. The project leverages **TensorFlow** and **transfer learning** models to enhance prediction accuracy using both textual and image data.

## Project Overview

Etsy is a global marketplace that connects millions of buyers and sellers. This project aims to predict various product attributes using advanced **deep learning** models and multimodal approaches, utilizing both text and image data to gain deeper insights into product characteristics.

### Objectives

- Predict product attributes (e.g., category, primary color, secondary color) based on product descriptions and metadata.
- Enhance search relevance and recommendation systems on Etsy by accurately predicting these attributes.
- Leverage **multimodal embedding extraction** using text embeddings from product descriptions and deep learning architectures.

## Features

1. **Multimodal Learning**: Utilizes both text-based and image-based data to predict product attributes.
2. **Attribute Prediction**: Models built to predict top category, primary color, and secondary color based on product descriptions.
3. **Transfer Learning**: Pretrained models from **TensorFlow Hub** are employed for both textual and image data processing to improve accuracy.

## Dataset

The dataset is a subset of Etsy's product listings, containing detailed product descriptions, images, and attributes such as top category, primary color, and secondary color.

- **Text Data**: Product titles and descriptions are processed through text embedding extraction to capture semantic relationships.
- **Image Data**: Images associated with product listings can be used for further multimodal exploration, though this project focuses primarily on text-based predictions.

## Methodology

1. **Data Preprocessing**:
   - Product descriptions are cleaned and tokenized.
   - Text embeddings are generated using pretrained models from **TensorFlow Hub**.
   - One-hot encoding is applied for labels such as category ID and color IDs.

2. **Model Architecture**:
   - A deep neural network (DNN) is implemented to predict product attributes based on text embeddings.
   - The architecture includes dense layers with activation functions tailored for the multi-class classification problem.

3. **Model Training**:
   - The model is trained using a **categorical cross-entropy loss function** and optimized with the **Adam optimizer**.
   - The dataset is split into training and validation sets to evaluate performance and avoid overfitting.

4. **Evaluation**:
   - **Accuracy** and **F1 score** are used as the primary metrics to assess model performance.

## Results

- **Top Category ID**: Achieved an F1 score of 0.74, indicating strong performance in categorizing products.
- **Primary Color ID**: Achieved an F1 score of 0.65.
- **Secondary Color ID**: F1 score of 0.63, with potential for improvement in distinguishing secondary colors.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/etsy-attribute-prediction.git
    ```

2. **Install Dependencies**:
    - Make sure you have Python installed, then install the required dependencies using pip:
      ```bash
      pip install -r requirements.txt
      ```

3. **Run the Jupyter Notebook**:
    To explore the model and run the experiments, open the provided `.ipynb` file:
    ```bash
    jupyter notebook Etsy_Jiaying.ipynb
    ```

## Technologies Used

- **Python**: The main programming language.
- **TensorFlow & TensorFlow Hub**: Used for model building and pre-trained embeddings.
- **Jupyter Notebook**: For running and documenting the experiments.
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: Used for evaluation metrics and additional preprocessing tasks.

## Future Work

- Explore **image-based models** to improve color prediction by incorporating product images.
- Experiment with more advanced **language models** for better text embedding extraction.
- Further fine-tuning of the model to enhance the prediction of secondary product attributes.

## License

This project is licensed under the MIT License.
