import {Form, Container, Button, Card} from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css';
import * as tf from '@tensorflow/tfjs';
import React from 'react';
// import {word_index} from 'https://raw.githubusercontent.com/dewball345/covid_text_sentiment_analysis/main/word_index.js';



function convert_to_tensors(word, word_index){
  word = word.split(" ")
  word = word.map(element => {
    if(word_index[element] !== undefined){
      return word_index[element]
    } else {
      return word_index[1]
    }
  });
  let n = 64-word.length;
  let padding = new Array(n); for (let i=0; i<n; ++i) padding[i] = 0;
  word = padding.concat(word)
  word = tf.tensor2d([word], [1, word.length])
  return word;
}

function indexOfMax(arr) {
  if(arr.length === 0){
    return -1
  }
  let arrindex = 0
  for(let index = 0; index < arr.length; index++ ){
    if(arr[index] > arr[arrindex]){
      arrindex = index
    } 
  }

  return arrindex
}

class App extends React.Component{

  constructor(){
    super()
    this.onSubmit = this.onSubmit.bind(this)
    this.state = {
      tweet: "",
      status: "",
      model:null,
      didLoadOnce: false,
      word_index: {}
    }
  }

  async componentDidMount(){
    
    // let json = JSON.parse('./tfjs_covid_text_class/model.json')
    // console.log(json)
    if(!this.state.didLoadOnce){
        let word_index = await fetch('https://raw.githubusercontent.com/dewball345/covid-text-analysis-2/main/word_index.json')
        word_index = await word_index.json();
        const model = await tf.loadLayersModel("https://raw.githubusercontent.com/dewball345/covid-text-analysis-2/main/model.json")
        // console.log(model.summary())
        this.setState({
          model: model,
          didLoadOnce:true,
          word_index:word_index
        })
    }
  }

  onSubmit(event){
    event.preventDefault();
    let input = convert_to_tensors(this.state.tweet, this.state.word_index);
    let classes = ["Sad", "Happy", "Little Sad", "Neutral", "Little Happy"]
    let prediction = Array(this.state.model.predict(input).dataSync())[0];
    let maxIndex = indexOfMax(prediction)
    this.setState({
      status:classes[maxIndex]
    });
    
  }

  render(){
      return (
        <Container className="d-flex" style={{
          minHeight:"100vh",
        }}>
          <Container className = "align-self-center" style={{
            maxWidth:"500px"
          }}>
            <Card className="shadow-lg" style={{
              borderRadius: "20px"
            }}>
              <Card.Header className="bg-primary" style={{
                borderTopLeftRadius: "20px",
                borderTopRightRadius: "20px",
                color: "white"
              }}>
                <Card.Title>
                    Covid-19 sentiment analysis
                </Card.Title>
                <Card.Subtitle style={{
                  fontWeight: 300
                }}>
                    Enter a tweet, and I will try to guess how positive/negative it is...
                </Card.Subtitle>
              </Card.Header>
              <Card.Body>
                <Form onSubmit = {this.onSubmit}>
                  <Form.Group>
                    <Form.Label>
                      <h5>
                        Enter a tweet
                      </h5>
                    </Form.Label>
                    <Form.Control type="text" placeholder="Enter text" onChange={(e) => {
                      this.setState({
                        tweet: e.target.value
                      });
                    }}/>
                  </Form.Group>

                  <Button className="btn-sm" variant="primary" type="submit">
                      Submit
                  </Button>
                </Form>
              </Card.Body>
              <Card.Footer className="bg-primary" style={{
                borderBottomLeftRadius: "20px",
                borderBottomRightRadius: "20px",
                color: "white"
              }}>
                <h4>Your tweet was: {this.state.status}</h4>
              </Card.Footer>
            </Card>
          </Container>
        </Container>
      );
  }
}

export default App;
