import uuid
import requests
import unittest

#Defining the main class: Process
class Process(object):
    def __init__(self):
        self.id=uuid.uuid1()
        self.predictions=[]
        self.models=[]

    def train_new_model(self,training_input):
        mod= Model(training_input)
        self.models.append(mod)
    
    def list_models(self):
        return self.models
        
    def list_predictions(self):
        return self.predictions

    def predict(self,model_id,predict_input):
        # a code that predicts the output based on the model, example a function called output()
        predict_output = output(model_id,predict_input)
        pred = Prediction(model_id,predict_input,predict_output)
        self.predictions.append(pred)



# Defining Model class, an attribute of Process
class Model(object):
    def __init__(self,training_input):
        self.id=uuid.uuid1()
        self.training_input=training_input
        # training code goes here {...}

# Defining Prediction class, an attribute of Process

class Prediction(object):
    def __init__(self,id_model,input,output):
        self.id=uuid.uuid1()
        self.trained_model=id_model
        self.input=input
        self.output=output


def output(a,b):
    return 'magic'

class TestConsumer(unittest.TestCase):

    def test_consume_food_consumes_the_apple(self):
        c = Process()
        
        self.assertTrue(c.apple.consumed,
                        "Expected apple to be consumed")

    def test_consume_food_cuts_the_food(self):
        c = Consumer()
        c.consume_food()
        self.assertTrue(c.apple.been_cut,
                        "Expected apple to be cut")

    def test_pick_food_always_selects_the_apple(self):
        c = Consumer()
        food = c.pick_food()
        self.assertEquals(c.apple, food,
                          "Expected apple to have been picked")

if __name__ == '__main__':

    # people = requests.get('http://api.open-notify.org/astros.json')

    # people_json  = people.json()

    # # print("Number of people in space:",people_json['number'])

    # # for p in people_json['people']:
    # #     print(p['name'])

    # parameter = {"rel_rhy":"jingle"}
    # request = requests.get('https://api.datamuse.com/words',parameter)

    # js=request.json()
    # for item in js:
    #     print(item['word'])
        # if __name__ == '__main__':
    unittest.main()