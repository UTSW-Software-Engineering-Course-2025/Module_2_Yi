#bug 0, cause an exception to when run
def fReplaceText(strInput, strText, intStart, intStop):
    """
    Replace the text between intStart and intStop with strText
    """
    #copy the string
    strReturn = strInput[:]
    #update the section of the string with inputted text
    strReturn = list(strReturn)
    strText = list(strText)
    for j in range(len(strText)):
        strReturn.insert(intStop+j, strText[j])
    strReturn.pop()
    # strReturn = ''.join(strReturn)
    del strReturn[intStart:intStop]

    
    strReturn = ''.join(strReturn)
    return str(strReturn)

def fFindAndReplaceInStr(strInput, strOld, strNew):
    """
    replace strOld with strNew in strInput
    """
    intLenOld = len(strOld)
    intStart = strInput.find(strOld)
    return fReplaceText(strInput, strNew, intStart, intStart+intLenOld)



#Hint: run the file in debug mode, VSCode should catch the exception.
#Then try the Error line of code in DEBUG CONSOLE, check these strings, how to make it right?
def main():
    print(fFindAndReplaceInStr("Luke, I am your father.", 'Luke', 'No'))

if __name__ == "__main__":
    main()