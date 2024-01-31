import phonenumbers
from phonenumbers import timezone
from phonenumbers import geocoder, carrier
from phonenumbers import is_valid_number

phoneNumber = phonenumbers.parse(input("Type here the Target phone number: "))

timeZone = timezone.time_zones_for_number(phoneNumber)

Carrier = carrier.name_for_number(phoneNumber, 'en')

Region = geocoder.description_for_number(phoneNumber, 'en')

if is_valid_number(phoneNumber):
  print(timeZone, Region)
  print(Carrier)
  
else:
  print("Invalid phone number")
