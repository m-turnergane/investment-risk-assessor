# Investment Risk Tolerance Assessment Tool

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Application Structure](#application-structure)
7. [Customization](#customization)
8. [Disclaimer](#disclaimer)
9. [Contact](#contact)
10. [License](#license)

## Introduction

This Investment Risk Tolerance Assessment Tool is a Streamlit-based web application designed to help individuals understand their risk tolerance when it comes to investments. By answering a series of questions about their financial situation, investment experience, and attitudes towards risk, users receive a personalized risk tolerance assessment and investment allocation recommendation.

Try it yourself!: https://investment-risk-assessment.streamlit.app/

## Features

- **User-friendly Interface**: Built with Streamlit for a smooth, interactive experience.
- **Comprehensive Questionnaire**: Covers various aspects of financial life, including:
  - Personal information (age, marital status, dependents)
  - Employment and income
  - Assets and liabilities
  - Investment experience
  - Risk attitudes and preferences
- **Dynamic Risk Assessment**: Utilizes both rule-based and machine learning models to calculate risk tolerance.
- **Personalized Results**: Provides a risk tolerance category and recommended asset allocation.
- **Data Persistence**: Saves anonymized user profiles for future analysis and model improvement.
- **Responsive Design**: Adapts to different screen sizes for desktop and mobile use.
- **Progress Tracking**: Shows users their progress through the assessment.
- **Back and Start Over Options**: Allows users to navigate back or restart the assessment.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Altair
- Scikit-learn
- Joblib
- SQLite3

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/m-turnergane/investment-risk-assessor.git
   cd risk-assessment-tool
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run risk_assessment_app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Follow the on-screen instructions to complete the risk assessment questionnaire.

4. Review your personalized risk tolerance assessment and investment allocation recommendation.

## Application Structure

- `risk_assessment_app.py`: Main application file containing the Streamlit interface and logic.
- `user_profiles.db`: SQLite database for storing anonymized user profiles.
- `user_data.csv`: CSV file for storing data used in the machine learning model.
- `risk_assessment_model.joblib`: Serialized machine learning model (created after sufficient data is collected).

## Customization

You can customize various aspects of the application:

- Modify the `questions` list in `risk_assessment_app.py` to add, remove, or change questions.
- Adjust the risk scoring logic in the `calculate_risk_score_rule_based` function.
- Modify the `get_risk_tolerance` function to change how risk scores map to risk tolerance categories.
- Update the `get_allocation` function to adjust the recommended asset allocations for each risk tolerance level.


## Future Plans and Potential Improvements

We're always looking to enhance the Investment Risk Tolerance Assessment Tool. Here are some areas we're considering for future development:

1. **Enhanced Machine Learning Model**: Improve the ML model's accuracy by incorporating more sophisticated algorithms and additional relevant features.

2. **Multi-language Support**: Implement internationalization to make the tool accessible to non-English speakers.

3. **Detailed Investment Recommendations**: Provide more specific investment suggestions based on the user's risk profile, potentially including specific asset classes or example portfolios.

4. **Integration with Financial Data APIs**: Incorporate real-time market data to provide more context-aware risk assessments and recommendations.

5. **User Accounts and Progress Tracking**: Allow users to create accounts, save their progress, and track changes in their risk tolerance over time.

6. **Mobile App Version**: Develop a native mobile application for iOS and Android platforms.

7. **Customizable Questionnaires**: Allow financial advisors or institutions to customize the questionnaire for their specific needs.

8. **Interactive Educational Content**: Add resources to help users understand investment concepts and the implications of different risk tolerances.

9. **Improved Data Visualization**: Enhance the results section with more interactive and informative charts and graphs.

10. **API Development**: Create an API for the risk assessment tool, allowing it to be integrated into other financial planning applications.

We welcome contributions and suggestions from the community. If you have ideas for improvements or would like to work on any of these features, please feel free to open an issue or submit a pull request on our GitHub repository.

## Disclaimer

This Investment Risk Tolerance Assessment Tool is for educational and informational purposes only. It does not constitute financial advice, and should not be relied upon as the sole basis for any investment decision. Always consult with a qualified financial advisor before making investment decisions.

The risk tolerance assessment and investment recommendations provided by this tool are based on the information you provide and general principles of investment risk. Your actual risk tolerance may differ, and appropriate investment strategies can vary based on individual circumstances not captured by this assessment.

## Contact

Muhammad Turner Gane
Email: [m.turnergane@gmail.com]
LinkedIn: [https://www.linkedin.com/in/muhammad-gane/]
GitHub: [https://github.com/m-turnergane]

Feel free to contact me with any questions, suggestions, or feedback about this project.
