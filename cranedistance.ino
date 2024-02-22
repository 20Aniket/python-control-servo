int trig = 12;
int echo = 10;
int redPin = 3;
int greenPin = 5;

void setup()
{
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  
  long duration, inches;
  duration = pulseIn(echo, HIGH);
  inches = duration / 74 / 2;
  Serial.println(inches);

  // Check distance and control the RGB LED
  if (inches < 12) {
    // If the distance is below 12 cm, turn the LED red
    analogWrite(redPin, 0);     // No intensity for red
    analogWrite(greenPin, 255); // Full intensity for green
    
  } else {
    // If the distance is 12 cm or above, turn the LED green
    analogWrite(redPin, 255);   // Full intensity for red
    analogWrite(greenPin, 0);   // No intensity for green
  }
  delay(500);
}