// Segment pins (a, b, c, d, e, f, g)
int segmentPins[] = {6, 7, 8, 9, 10, 11, 12};
// Digit control pins
int digitPins[] = {2, 3};

// Digit patterns for numbers 0-9 on a 7-segment display
byte digits[10][7] = {
  {LOW, LOW, LOW, LOW, LOW, LOW, HIGH},    // 0
  {HIGH, LOW, LOW, HIGH, HIGH, HIGH, HIGH}, // 1
  {LOW, LOW, HIGH, LOW, LOW, HIGH, LOW},   // 2
  {LOW, LOW, LOW, LOW, HIGH, HIGH, LOW},   // 3
  {HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW},  // 4
  {LOW, HIGH, LOW, LOW, HIGH, LOW, LOW},   // 5
  {LOW, HIGH, LOW, LOW, LOW, LOW, LOW},    // 6
  {LOW, LOW, LOW, HIGH, HIGH, HIGH, HIGH}, // 7
  {LOW, LOW, LOW, LOW, LOW, LOW, LOW},     // 8
  {LOW, LOW, LOW, LOW, HIGH, LOW, LOW}     // 9
};

void setup() {
  Serial.begin(9600); // Start serial communication for debugging

  // Initialize all segment pins as OUTPUT
  for (int i = 0; i < 7; i++) {
    pinMode(segmentPins[i], OUTPUT);
  }

  // Initialize all digit control pins as OUTPUT
  for (int i = 0; i < 2; i++) {
    pinMode(digitPins[i], OUTPUT);
  }

  // Turn off all digits initially
  for (int i = 0; i < 2; i++) {
    digitalWrite(digitPins[i], LOW);
  }
}

void loop() {
  // Read the analog input (0-1023)
  int sensorValue = analogRead(A0);

  // Convert sensor value to voltage (Assuming 5V reference)
  float voltage = sensorValue * (5.0 / 1023.0);

  // Get the integer part of the voltage
  int displayValue = int(voltage); // Only keep the integer part
  int firstDigit = displayValue / 10;  // Tens place (if applicable)
  int secondDigit = displayValue % 10; // Ones place

  // Print voltage to the Serial Monitor for debugging
  Serial.print("Voltage: ");
  Serial.print(voltage, 2); // Display the actual voltage with 2 decimal places in Serial Monitor
  Serial.println(" V");

  // Display the first digit (if non-zero, for tens place)
  if (displayValue >= 10) {
    displayDigit(firstDigit, 0);
    delay(5); // Short delay for multiplexing
  }

  // Display the second digit (ones place)
  displayDigit(secondDigit, 1);
  delay(5); // Short delay for multiplexing
}

// Function to display a single digit on a specified position
void displayDigit(int digit, int position) {
  // Turn off both digits
  for (int i = 0; i < 2; i++) {
    digitalWrite(digitPins[i], LOW);
  }

  // Set the segments for the digit
  for (int i = 0; i < 7; i++) {
    digitalWrite(segmentPins[i], digits[digit][i]);
  }

  // Enable the specified digit
  digitalWrite(digitPins[position], HIGH);
}