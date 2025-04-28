const byte pinA = 3; // INT0
const byte pinB = 2; // INT1

volatile int encoderCount = 0;
volatile byte prevState = 0;

void setup() {
  pinMode(pinA, INPUT);
  pinMode(pinB, INPUT);
  attachInterrupt(digitalPinToInterrupt(pinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pinB), updateEncoder, CHANGE);
  
  Serial.begin(1000000);
  
  prevState = (digitalRead(pinA) << 1) | digitalRead(pinB);
}

void loop() {
  Serial.println(encoderCount);
}

void updateEncoder() {
  byte a = digitalRead(pinA);
  byte b = digitalRead(pinB);
  byte currState = (a << 1) | b;

  // State transition table
  // Index = (prevState << 2) | currState
  const int8_t transitionTable[16] = {
     0,  -1,   1,   0,
     1,   0,   0,  -1,
    -1,   0,   0,   1,
     0,   1,  -1,   0
  };

  int8_t direction = transitionTable[(prevState << 2) | currState];
  encoderCount += direction;
  prevState = currState;
}