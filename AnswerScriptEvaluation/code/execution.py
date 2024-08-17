from preprocessing import *
from extracting import *


def PPandE(image_file):

    GRAYSCALE(image_file)
    BLACKNWHITE(image_file)
    NOICEREMOVAL(image_file)

    text = EXTRACT(Image.open(image_file))

    return text



def main():

    qpaper = "qpaper.jpg"
    akey = "akey.jpg"
    asheet = "asheet.jpg"


    question_paper = PPandE(qpaper)
    answer_key = PPandE(akey)
    answer_sheet = PPandE(asheet)

    print(question_paper)
    print(answer_key)
    print(answer_sheet)
    import os

    os.environ["REPLICATE_API_TOKEN"] = "r8_QmLhI2bP5YFpei54svfzvzOGuMNzilo0BlZ7f"

    import replicate

    # Prompts
    prompt = '''You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.
You will be given 3 pieces of information: the question paper, the answer key, and the student's answer sheet.
Your task is to calculate the final score of the student based on the following rules and regulations:
The number mentioned after the question is the number of marks allocated for that question.
The number before the question represents the question number.
Calculate the percentage similarity of the student's answer compared to the answer key for each question.
Allocate marks based on the percentage similarity:
If the percentage is greater than 74%, allocate full marks for that question.
If the percentage is between 50%, and 74% (inclusive), allocate floor(3 * (question marks / 4)) marks.
If the percentage is between 25%, and 49% (inclusive), allocate floor(question marks / 2) marks.
If the percentage is less than 25%, allocate 0 marks.
Sum the total marks obtained by the student and the total marks available in the question paper.
Output only these two values: the total student score and the total question paper score. Do not include any other information.
Ensure the calculations are precise and adhere strictly to the rules provided.
'''

    qp = "now this is the question paper:"

    ak = "and this is the answer key:"

    ash = "and this is the student's answer sheet:"

    # Generate LLM response
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                            input={"prompt": f"{prompt}{qp}{question_paper}{ak}{answer_key}{ash}{answer_sheet} Assistant: ", # Prompts
                            "temperature":0.1, "top_p":0.9, "max_length":128, "repetition_penalty":1})  # Model parameters

    full_response = ""

    for item in output:
        full_response += item

    print(full_response)



if __name__ == "__main__":
    main()