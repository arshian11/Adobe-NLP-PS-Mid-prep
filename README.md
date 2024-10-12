# 💬 **Problem Description**

The process of communication is defined by marketing researchers as below:
A receiver, upon receiving a message from a sender over a channel, interacts with the
message, thereby generating effects (user behavior). Any message is created to serve an end
goal. For a marketer, the eventual goal is to get the desired effect (user behavior) i.e. such as
likes, comments, shares and purchases, etc.

![](./assests/Capture2.JPG)

In this challenge, we will try to solve the problem of behavior simulation (Task-1) and content
simulation (Task-2), thereby helping marketers to estimate user engagement on their social
media content as well as create content that elicits the desired key performance indicators (KPI)
from the audience.

---

# :books: **Dataset**

Brands use Twitter to post marketing content about their products to serve several purposes,
including ongoing product campaigns, sales, offers, discounts, brand building, community
engagement, etc. User engagement on Twitter is quantified by metrics like user likes, retweets,
comments, mentions, follows, clicks on embedded media and links. For this challenge, we have
sampled tweets posted in the last five years from Twitter enterprise accounts. Each sample
contains tweet ID, company name, username, timestamp, tweet text, media links and user
likes.

![](./assests/Capture3.JPG)

## :microscope: Data Visualization

The content field alone is not sufficient to grasp the user behaviour so we create a new field
comprised of date,content and infered company called formatted_text. The full image URLs has also
been extracted in a new field to make it easier accessing the images.

![](./assests/Capture4.JPG)

The formatted_text field will be tokenized to be later used by the model, we need to understand
the token length distribution so that we can create a uniform token length for the model to use.

![](./assests/download.png)

# 🌟 **Solution**
Task-1: Behavior Simulation
- Given the content of a tweet (text, company, username, media URLs, timestamp), the
task is to predict its user engagement, measured by likes.

## Approach :one:

Here we employ the use of DistilBERT a smaller and faster version of BERT.
We load a pre-trained model from the transfromers library.<br>

- The model outputs the hidden states, from which you select the embedding corresponding to the [CLS] token (first token).<br>
- After obtaining embeddings from BERT, a small feedforward neural network (fully connected layers) is used to predict the number of likes.<br>
- A Linear layer that transforms the hidden states (DistilBERT output) to a size of 180. The role of this layer is to reduce the dimensionality of the BERT embedding while retaining useful information. 

```python
 nn.Linear(self.bert.config.hidden_size, 180),
 nn.ReLU(),
 nn.Linear(180, 1)  # Regression output
```
1. DistilBERT hidden state → 768 dimensions (CLS token)
2. First Linear Layer → 180 dimensions
3. ReLU Activation → Adds non-linearity
4. Second Linear Layer → 1 dimension (predicted likes)

- The loss function used is Mean Squared Error (MSE) because it's a regression task

## Approach :two:

In the previous approach we only utilised the formatted_text field to predict the no. of likes but as we can see we are leaving the media field which might contain usedful information
pertaining to the no. of likes in a tweet.<br>
For this task we use a CLIP model.

The model expects images to be of size 224x224, so we load the image from the URL and resize it

```python
def load_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((224, 224))
        return image
    else:
        return create_random_image(224, 224)
```

In my testing experienxce I found out that many URLs are not responsive and break the training loop. To counter this challenge I opted to create a random image of the same size to bee given ot the model
other than simply a black image.

![](./assests/Capture5.JPG)

```python
def create_random_image(width, height):
    random_image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image_array)
    return random_image
```

