float sigmoid(float x){
  return 1.0/(1.0+exp(-x));
}

float sigmoid_derivative(float x){
  return x*(1.0-x);
}
  