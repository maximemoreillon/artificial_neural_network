NeuralNetwork neural_network;

Table training_inputs_table;
Table training_outputs_table;

float[][] training_inputs;
float[][] training_outputs;

float[][] validation_inputs;
float[][] validation_outputs;

int validation_input_index = 0;

int[] network_structure = new int[0]; // layers will be added later

void setup(){
  size(1280, 1000);
  
  training_data_init();
  
  validation_inputs = training_inputs;
  validation_outputs = training_outputs;
  
  // Network initialization
  network_structure = append(network_structure,5);
  network_structure = append(network_structure,training_outputs[0].length);
  neural_network = new NeuralNetwork(network_structure, training_inputs[0].length);
  
  
}

void draw(){
  background(0);
  
  neural_network.train(training_inputs,training_outputs,0.005);
  neural_network.forward_propagation(validation_inputs[validation_input_index]);
  neural_network.display(width/2, height/2, width/1.2, height/1.2, validation_inputs[validation_input_index]);
  
  int text_y = 50;
  fill(255);
  textAlign(LEFT,UP);
  textSize(16);
  text("Epoch: " + neural_network.epoch,100,text_y);
  text("Error: " + nf(100*neural_network.error/training_inputs.length,2,2) + "%",220,text_y);
  text("Validation input index: " + (validation_input_index+1),400,text_y);
  
  text("Artificial neural network visualization tool" ,width-350,height-50);
  text("Maxime MOREILLON" ,width-350,height-25);
  
}

void keyPressed(){
  
  if(keyCode == UP){
    validation_input_index ++;
    if(validation_input_index >= validation_inputs.length){
      validation_input_index = 0;
    }
  }
  else if (keyCode == DOWN){
    validation_input_index --;
    if(validation_input_index < 0){
      validation_input_index = validation_inputs.length-1;
    }
  }
  else if (key == ' '){
    neural_network = new NeuralNetwork(network_structure, training_inputs[0].length);
  }
  
}
