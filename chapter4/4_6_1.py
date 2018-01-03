import re

mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
print(mySent.split())

regEx = re.compile('\W*')
listOfTokens = regEx.split(mySent)
print(listOfTokens)

print([tok for tok in listOfTokens if len(tok) > 0])
print([tok.lower() for tok in listOfTokens if len(tok) > 0])

emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
