//Initializing all variables associated with analog sensor readings
int deltoidSensor;
int pecMajorSensor;
int pecMinorSensor;
int longHeadSensor;
int lateralHeadSensor;
bool ledOn;
int ledPin = 51;
int frame = 0;

//Initial setup, runs once
void setup() 
{  
  //Beginning serial output at baud rate of 115200
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
  delay(10000);
  digitalWrite(ledPin, HIGH);
  ledOn = true;
}

//Main code, runs on loop
void loop() 
{
  deltoidSensor = analogRead(A0);
  pecMajorSensor = analogRead(A1);
  pecMinorSensor = analogRead(A2);
  longHeadSensor = analogRead(A3);
  lateralHeadSensor = analogRead(A4);
  //Printing values using serial com (port 5)
  Serial.println(deltoidSensor);
  Serial.println(pecMajorSensor);
  Serial.println(pecMinorSensor);
  Serial.println(longHeadSensor); 
  Serial.println(lateralHeadSensor); 
  Serial.println(micros());
  Serial.println(frame);
  frame++;

  //200 ms delay to avoid overloading file
  delay(31);
}
