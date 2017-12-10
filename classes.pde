class NeuralNetwork {

  float[][][] synapses;
  float[][] neurons;
  float[][] deltas;
  
  int epoch;
  float cost;
  float error;

  NeuralNetwork(int[] network_structure, int network_input_count) {
    // Constructor

    synapses = new float[network_structure.length][][]; // defined as [layer][neuron][input]
    neurons = new float[network_structure.length][];
    deltas = new float[network_structure.length][];
    epoch = 0;

    for (int layer_index=0; layer_index<synapses.length; layer_index++) {

      int neuron_count = network_structure[layer_index];
      int input_count;
      if (layer_index == 0) {
        // First layer takes the network input as input
        input_count = network_input_count;
      } else {
        // Other layers take the neurons of the previous layer as input
        input_count = network_structure[layer_index-1];
      }

      synapses[layer_index] = new float[neuron_count][input_count+1]; // Adding bias
      neurons[layer_index] = new float[neuron_count];
      deltas[layer_index] = new float[neuron_count];

      // Synapses initialized randomly
      for (int neuron_index=0; neuron_index<synapses[layer_index].length; neuron_index++) {
        for (int input_index=0; input_index<synapses[layer_index][neuron_index].length; input_index++) {
          synapses[layer_index][neuron_index][input_index] = random(-1, 1);
        }
      }
    }
  }

  float sigmoid(float x) {
    return 1.0/(1.0+exp(-x));
  }

  float sigmoid_derivative(float x) {
    return x*(1.0-x);
  }

  void forward_propagation(float[] network_input) {
    // Computes the output of the network with a given input

      for (int layer_index=0; layer_index<synapses.length; layer_index++) {

      // Input is network output for first layer and then neurons of previous layer
      float[] input;
      if (layer_index == 0) input = network_input;
      else input = neurons[layer_index-1];

      for (int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index++) {
        float synaptic_sum = 0;
        // deal first with normal inputs of the layer
        for (int input_index=0; input_index < synapses[layer_index][neuron_index].length-1; input_index++) {
          synaptic_sum += input[input_index]*synapses[layer_index][neuron_index][input_index];
        }

        // deal with bias separately
        int bias_index = synapses[layer_index][neuron_index].length-1;
        synaptic_sum += synapses[layer_index][neuron_index][bias_index];
        
        // Activate each neuron
        neurons[layer_index][neuron_index] = sigmoid(synaptic_sum);
      }
    }
  }
  
  void train(float[][] training_inputs, float[][] training_outputs, float learning_rate) {
    // Adjusts the network's so as to yield a correct output when given a certain input
    
    cost = 0.0;
    error = 0.0;
    for (int training_set_index=0; training_set_index<training_inputs.length; training_set_index++) {

      float[] training_input = training_inputs[training_set_index];
      float[] training_output = training_outputs[training_set_index];

      forward_propagation(training_input);

      float[] network_output = neurons[neurons.length-1];
      
      // computing cost
      for (int output_index=0; output_index<training_output.length; output_index++) {
        error += abs(training_output[output_index] - network_output[output_index]);
        cost += 0.5 * (training_output[output_index] - network_output[output_index]) * (training_output[output_index] - network_output[output_index]);
      }

      backward_propagation(training_input, training_output);
      update_weights(training_input, learning_rate);
    }
    epoch ++;
  }

  void backward_propagation(float[] training_input, float[] training_output) {
    // Computes the delta of each neuron

    // Iterate backwards through the network
    for (int layer_index = synapses.length-1; layer_index >= 0; layer_index --) {
      for (int neuron_index=0; neuron_index < synapses[layer_index].length; neuron_index++) {

        float error;
        if (layer_index == synapses.length-1) {
          // Last layer
          error = training_output[neuron_index] - neurons[synapses.length-1][neuron_index];
        } else {
          // Layers other than the last one
          error = 0.0;
          for (int next_layer_neuron_index = 0; next_layer_neuron_index < synapses[layer_index+1].length; next_layer_neuron_index++) {
            error += deltas[layer_index+1][next_layer_neuron_index] * synapses[layer_index+1][next_layer_neuron_index][neuron_index];
          }
        }
        deltas[layer_index][neuron_index] = error * sigmoid_derivative(neurons[layer_index][neuron_index]);
      }
    }
  }

  void update_weights(float[] training_input, float learning_rate) {
    // Updates the synapses of the network using the deltas computed with backward_propagation

    for (int layer_index=0; layer_index<synapses.length; layer_index++) {
 
      // Use network input for the first layer and previouslayer neurons for the other layers
      float[] input;
      if (layer_index == 0) input = training_input;
      else input = neurons[layer_index-1];

      for (int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index++) {
        // Normal inputs
        for (int input_index = 0; input_index < synapses[layer_index][neuron_index].length-1; input_index++) {
          synapses[layer_index][neuron_index][input_index] += learning_rate * deltas[layer_index][neuron_index] * input[input_index];
        }

        // bias
        int bias_index = synapses[layer_index][neuron_index].length-1;
        synapses[layer_index][neuron_index][bias_index] += learning_rate * deltas[layer_index][neuron_index];
      }
    }
  }

  


  void display(float x, float y, float w, float h, float[] network_input) {
    // Display the neural network at (x,y) with a width w and height h and with its neurons activated according to the given input
    
    // Parameters
    float neuron_radius = 30;
    float neuron_stroke_weight = 3;

    float synapse_min_width = 0;
    float synapse_max_width = 4;
    float synapse_min_brightness = 20;
    float synapse_max_brightness = 255;

    // synapses
    stroke(255);
    noFill();
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {

      // find min and max weights of current layer
        // THIS SHOULD BE CHANGED TO THE FIRST SYNAPSE
      float min_weight = 9999;
      float max_weight = -9999;
      for (int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index++) {
        for (int input_index = 0; input_index < synapses[layer_index][neuron_index].length; input_index++) {
          if (synapses[layer_index][neuron_index][input_index] > max_weight) {
            max_weight = synapses[layer_index][neuron_index][input_index];
          }
          if (synapses[layer_index][neuron_index][input_index] < min_weight) {
            min_weight = synapses[layer_index][neuron_index][input_index];
          }
        }
      }

      float startX = map(layer_index-1, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      float endX = map(layer_index, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      
      for (int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index++) {
        
        float endY;
        if(layer_index == synapses.length-1) {
           endY = map(neuron_index, -1, synapses[layer_index].length, y-0.5*h, y+0.5*h);
        }
        else {
          // The last neuron of the layer is the bias, which is not connected to the previous layer
          endY = map(neuron_index, -1, synapses[layer_index].length+1, y-0.5*h, y+0.5*h);
        }
        
        for (int input_index=0; input_index<synapses[layer_index][neuron_index].length; input_index++) {
          float startY = map(input_index, -1, synapses[layer_index][neuron_index].length, y-0.5*h, y+0.5*h);

          float synapse_weight = synapses[layer_index][neuron_index][input_index];
          float colorMap = 0;
          float widthMap = 0;
          if (synapse_weight>0) {
            colorMap = map(synapse_weight, 0, max_weight, synapse_min_brightness, synapse_max_brightness);
            widthMap = map(synapse_weight, 0, max_weight, synapse_min_width, synapse_max_width);
            stroke(colorMap, 0, 0);
            strokeWeight(widthMap);
          }
          else {
            colorMap = map(synapse_weight, min_weight, 0, synapse_max_brightness, synapse_min_brightness);
            widthMap = map(synapse_weight, min_weight, 0, synapse_max_width, synapse_min_width);
            stroke(colorMap);
            strokeWeight(widthMap);
          }

          line(startX, startY, endX, endY);
        }
      }
    }

    // Neurons
    ellipseMode(RADIUS);
    textAlign(CENTER, CENTER);
    textSize(10);
    strokeWeight(neuron_stroke_weight);
    fill(0);
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {
      float pos_x = map(layer_index, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      for (int neuron_index = 0; neuron_index < neurons[layer_index].length; neuron_index++) {
        float pos_y;
        if(layer_index == synapses.length-1) {
           pos_y = map(neuron_index, -1, synapses[layer_index].length, y-0.5*h, y+0.5*h);
        }
        else {
          pos_y = map(neuron_index, -1, synapses[layer_index].length+1, y-0.5*h, y+0.5*h);
        }
        
        float col = map(neurons[layer_index][neuron_index], 1, 0, 0, 255);
        stroke(255, col, col);
        fill(0);
        ellipse(pos_x, pos_y, neuron_radius, neuron_radius);
        fill(255, col, col);
        text(neurons[layer_index][neuron_index], pos_x, pos_y);
      }
    }

    // Bias
    ellipseMode(RADIUS);
    textAlign(CENTER, CENTER);
    strokeWeight(neuron_stroke_weight);
    textSize(10);
    fill(0);
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {
      float pos_x = map(layer_index-1, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      float pos_y = map(synapses[layer_index][0].length-1, -1, synapses[layer_index][0].length, y-0.5*h, y+0.5*h);  // not too sure about this one
      stroke(255, 0, 0);
      fill(0);
      ellipse(pos_x, pos_y, neuron_radius, neuron_radius);
      fill(255, 0, 0);
      text("1", pos_x, pos_y);
    }
    
    
    
    
    // Inputs
    // find min and max weights of current layer
    float min_input = 9999;
    float max_input = -9999;
    for (int input_index = 0; input_index < network_input.length; input_index++) {
      if (network_input[input_index] > max_input) {
        max_input = network_input[input_index];
      }
      if (network_input[input_index] < min_input) {
        min_input = network_input[input_index];
      }
    }
      
      
    rectMode(CENTER);
    strokeWeight(neuron_stroke_weight);
    fill(0);
    for (int input_index=0; input_index<network_input.length; input_index++) {
      float pos_x = x-0.5*w;
      float pos_y = map(input_index, -1, network_input.length+1, y-0.5*h, y+0.5*h);
      float colorMap = 0;
      float input = network_input[input_index];
      if (input>0) {
        colorMap = map(input, 0, max_input, synapse_min_brightness, synapse_max_brightness);
        stroke(colorMap, 0, 0);
      }
      else {
        colorMap = map(input, min_input, 0, synapse_max_brightness, synapse_min_brightness);
        stroke(colorMap);
      }
      fill(0);
      rect(pos_x, pos_y, 2*neuron_radius, 2*neuron_radius);
      fill(255);
      text(input, pos_x, pos_y);
    }
  }
}