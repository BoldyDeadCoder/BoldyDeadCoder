import phonenumbers
from phonenumbers import timezone
from phonenumbers import geocoder, carrier
from phonenumbers import is_carrier_specific_for_region

phoneNumber = phonenumbers.parse(input("Type here the Target phone number: "))

timeZone = timezone.time_zones_for_number(phoneNumber)

Carrier = carrier.name_for_number(phoneNumber)

Carrier_is_specific_for_carrier_region = is_carrier_specific_for_region(.parse."US",(phoneNumber)

Region = geocoder.description_for_number(phoneNumber)

if phoneNumber.is_valid_number(phoneNumber):
  print(timeZone, Region,)
  print(Carrier_is_specific_for_carrier_region)
  
elif phoneNumber.is_invalid_number(phoneNumber):
  phoneNumber = phonenumbers.parse(input("Type here the Target phone number: "))
  print(timeZone, Region,)
  print(Carrier_is_specific_for_carrier_region)
  print(phoneNumber.is_valid_number(phoneNumber))

print(phoneNumber.is_valid_number(phoneNumber)


