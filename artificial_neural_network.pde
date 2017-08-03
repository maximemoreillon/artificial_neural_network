Table training_inputs_table, training_outputs_table;

NeuralNetwork neural_network;



float[][] training_inputs; // = {{2.7810836,2.550537003},{1.465489372,2.362125076},{3.396561688,4.400293529},{1.38807019,1.850220317},{3.06407232,3.005305973},{7.627531214,2.759262235},{5.332441248,2.088626775},{6.922596716,1.77106367},{8.675418651,-0.242068655},{7.673756466,3.508563011}};
float[][] training_outputs; // = {{0},{0},{0},{0},{0},{1},{1},{1},{1},{1}};
float[][] test_inputs; // = training_inputs;

int test_input_index = 0;

int[] network_structure = {10, 2};

void setup(){
  size(1280,1024);
  
  training_inputs_table = loadTable("training_inputs.csv");
  training_outputs_table = loadTable("training_outputs.csv");
  
  // initialize arrays
  training_inputs = new float[training_inputs_table.getColumnCount()][training_inputs_table.getRowCount()];
  training_outputs = new float[training_outputs_table.getColumnCount()][training_outputs_table.getRowCount()];
  
  for (int column=0; column<training_inputs_table.getColumnCount(); column++) {
    for (int row=0; row<training_inputs_table.getRowCount(); row++){
      training_inputs[column][row] = training_inputs_table.getFloat(row,column);
    }
  }
  
  for (int column=0; column<training_outputs_table.getColumnCount(); column++) {
    for (int row=0; row<training_outputs_table.getRowCount(); row++){
      training_outputs[column][row] = training_outputs_table.getFloat(row,column);
    }
  }
  
  test_inputs = training_inputs;


  neural_network = new NeuralNetwork(network_structure, training_inputs[0].length);
  
  
}

void draw(){
  background(0);
  
  neural_network.train(training_inputs,training_outputs,0.5);
  neural_network.display(width/2, height/2, width/1.2, height/1.2, test_inputs[test_input_index]);
  
  fill(255);
  text(test_input_index+1,20,20);
}

void keyPressed(){
  
  if(keyCode == UP){
    test_input_index ++;
    if(test_input_index >= test_inputs.length){
      test_input_index = 0;
    }
  }
  else if (keyCode == DOWN){
    test_input_index --;
    if(test_input_index < 0){
      test_input_index = test_inputs.length-1;
    }
  }
  else if (key == ' '){
    neural_network = new NeuralNetwork(network_structure, training_inputs[0].length);
  }
  
}