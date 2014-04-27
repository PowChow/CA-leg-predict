from pymongo import MongoClient
import sunlight
import json
import pprint

#=======================================================================================
# CONNECTION & API KEY's
#=======================================================================================

sunlight.config.API_KEY = "somelettersandnumbers"
sunlight.config.API_SIGNUP_PAGE = 'http://sunlightfoundation.com/api/accounts/register/'
sunlight.config.KEY_ENVVAR = 'SUNLIGHT_API_KEY'
sunlight.config.KEY_LOCATION = '~/.sunlight.key'

#=======================================================================================
# Url with user and password to MongoHQ database
#=======================================================================================

MONGOHQ_URL = 'mongodb://USER:PASSWORD@oceanic.mongohq.com:10036/openstates'

'''
@param param: None
@note: connect to MongoDB database and return database object
'''
def EstablishConnection():
    
    client  = MongoClient(MONGOHQ_URL)      #establish connection to database 
    db = client.openstates                  #connect to 'openstates' database
    
    return db

#Check connection to MongoDB
print "Yes, you have a connection"

#=======================================================================================
# FUNCTIONS 
#=======================================================================================
     
def PostDB(data, table):
    '''
    @param bill: Json object containing data into table
    @param bill_table: table object to store info back in database
    @note: originally function tried to deal with keys for Json data not in unicode format...encode these values and skip errors
    '''   
    table.insert(d)          

#=======================================================================================
# CALL SUNLIGHT API & PUSH TO DATABASE
#=======================================================================================

#List of states for which data will be obtained

# STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
#           "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
#           "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
#           "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
#           "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

STATES = ["CA"]


def main():
    #============================================================================
    #Establish connection and make database object
    #============================================================================
    db  = EstablishConnection()
    
    bill_table = db.ca_bills               #bill table
    bill_d_table = db.bills_details        #bill details table
    legislator_table = db.legislators      #legislator table
    committee_table = db.committees        #committee table
    

    #============================================================================
    #Get data from bills, legislators, and committees
    #============================================================================
    
    #Getting bills using fields: state, first_name, last_name, chamber, state
    #                            active (true or false, true default), term,
    #                            district, party 
    
    #When data from all states is necessary, loop over list of states
    #for state in STATES:
    #    bills_data = sunlight.openstates.bills(state = state, chamber = 'lower')

    #bills_data = sunlight.openstates.bills(state = state, chamber = 'lower')
    #bills_data = sunlight.openstates.bills(state = state, chamber = 'upper')
    #committee_data = sunlight.openstates.committees()
    #legislators_data = sunlight.openstates.legislators()
    #committee_data = sunlight.openstates.committees()
    
    #============================================================================
    #Send data to database
    #============================================================================
    
    #PostDB(bills_data,bill_table)
    #PostDB(legislators_data, legislator_table)
    #PostDB(committee_data, committee_table)


    # From MongoDB: Created list of dicts, where dict keys are bill_id & list of sessions  
    # Getting list of tuples for API call to bill details
    bill_session = bill_table.aggregate(
        { '$group': {
             '_id': '$bill_id', 
             'session': {'$push':'$session'}}
        }).values()[1]
 
  # list_bill_session = [ list_bill_session.append(str(bill_session[b].values()[1]), list(bill_session[b].values()[0])) for b in bill_session ]
    list_bill_session =[]

    for i, entry in enumerate(bill_session):
        b_id = str(bill_session[i].values()[1]) 
        for s in xrange(1, len(bill_session[i].values()[0])):
            b_session = str(bill_session[i].values()[0][s])
            list_bill_session.append((b_id, b_session))
        
    #API calls for bill details using unique list of bill_id & session
    #example: sunlight.openstates.bill_detail(state="CA",session='20092010', bill_id='SCR 2')
    list_bill_details = [ sunlight.openstates.bill_detail(state='CA',session=session, bill_id=bill) for bill, session in list_bill_session  ]
    bill_d_table.insert(bill_details)


if __name__ == '__main__':
    main()

print("\n=======================Sucess!=====================================")




                
