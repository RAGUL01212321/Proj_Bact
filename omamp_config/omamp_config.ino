float previousVoltage = 0;
unsigned long previousTime = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  // Read actual voltage from A0
  float voltage = analogRead(A0) * (5.0 / 1023.0);
  unsigned long currentTime = millis();

  if (previousTime > 0) {
    float growthRate = (voltage - previousVoltage) / ((currentTime - previousTime) / 1000.0);

    // Print the output
    Serial.print("Bacterial Growth Signal: ");
    Serial.print(voltage, 2);
    Serial.print(" V, Growth Rate: ");
    Serial.print(growthRate, 4);
    Serial.println(" V/s");
  }

  previousVoltage = voltage;
  previousTime = currentTime;
  delay(1000);
}
