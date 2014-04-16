from pymongo import MongoClient
import sunlight
import json
import pprint

#=======================================================================================
# CONNECTION & API KEY's
#=======================================================================================

sunlight.config.API_KEY = "f89f5a1b63654d02ab53b5295feac047"
sunlight.config.API_SIGNUP_PAGE = 'http://sunlightfoundation.com/api/accounts/register/'
sunlight.config.KEY_ENVVAR = 'SUNLIGHT_API_KEY'
sunlight.config.KEY_LOCATION = '~/.sunlight.key'

#=======================================================================================
# Url with user and password to MongoHQ database
#=======================================================================================

MONGOHQ_URL = 'mongodb://pchow:gadsla2014@oceanic.mongohq.com:10036/openstates'

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

'''
@param e: value of key in json object
@note: reads each element e and makes a list from it 
'''
def ConvertToList(e):
    lst = []
    for x in e:
        lst.append(x)
    return lst

'''
@param bill: Json object containing bill data
@param bill_table: table object to store info back in database
@note: some keys for Json data are not in unicode format...need to figure out 
       whether to discard element or assign null value
'''        
def PostBillInfoToDatabase(bills, bill_table):
    for b in bills:
        try:
            bill_table.insert({
                               'title':b['title'],
                               'state':b['state'],
                               'session':b['session'],
                               'bill_id':b['bill_id'],
                               'type':b['type'],
                               'chamber':b['chamber'],
                               'subjects':b['subjects'],
                               'type':b['type'],
                               'id':b['id'],
                               })
        except:
            print "UnicodeEncodeError: 'ascii' codec can't encode characters......will not be sending this element to database"
            

'''
@param bill: Json object containing bill details data
@param bill_table: table object to store info back in database
'''        
def PostBillDetailsInfoToDatabase(bills_d, bill_d_table):
    for b in bills_d:
        try:
            bill_d_table.insert(bills_d)
        except:
            print "UnicodeEncodeError: 'ascii' codec can't encode characters......will not be sending this element to database"
            
'''
@param legislator: Json object containing bill data
@param legislator_table: table object to store info back in database
@note: some keys for Json data are not in unicode format...need to figure out 
       whether to discard element or assign null value
'''            
def PostLegislatorInfoToDatabase(legislator, legislator_table):
    for l in legislator:
        try:
            legislator_table.insert({
                                     'active':l['active'],
                                     'all_ids':ConvertToList(l['all_ids']),
                                     'chamber':l['chamber'],
                                     'district':l['district'],
                                     'full_name':l['full_name'],
                                     'leg_id':l['leg_id'],
                                     'party':l['party'],
                                      'state':l['state']
                                     })
        except:
            print "UnicodeEncodeError: 'ascii' codec can't encode characters......will not be sending this element to database"
            
'''
@param committee: Json object containing bill data
@param committee_table: table object to store info back in database
@note: some keys for Json data are not in unicode format...need to figure out 
       whether to discard element or assign null value
'''            
def PostCommitteeInfoToDatabase(committee, committee_table):
    for c in committee:
        try:
            committee_table.insert({
                                    'all_ids':ConvertToList(c['all_ids']),
                                    'chamber':c['chamber'],
                                    'committee':c['committee'],
                                    'id':c['id'],
                                    'parent_id':c['parent_id'],
                                    'state':c['state'],
                                    'subcommittee':c['subcommittee']
                                    })
        except:
            print "UnicodeEncodeError: 'ascii' codec can't encode characters......will not be sending this element to database"

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
    
    bill_table = db.bills                  #bill table
    bill_d_table = db.bills_details        #bill details table
    legislator_table = db.legislators      #legislator table
    committee_table = db.committees        #committee table
    

    #============================================================================
    #Get data from bills, legislators, and committees
    #============================================================================
    
    #Getting bills using fields: state, first_name, last_name, chamber, state
    #                            active (true or false, true default), term,
    #                            district, party 
    
    #for state in STATES:
    #    bills_data = sunlight.openstates.bills(state = state, chamber = 'lower')

    #bills_data = sunlight.openstates.bills(state = state, chamber = 'upper')
    #committee_data = sunlight.openstates.committees()
    #legislators_data = sunlight.openstates.legislators()
    #committee_data = sunlight.openstates.committees()
    
    #============================================================================
    #Send data to database
    #============================================================================
    
    #PostBillInfoToDatabase(bills_data,bill_table)
    #PostLegislatorInfoToDatabase(legislators_data, legislator_table)
    #PostCommitteeInfoToDatabase(committee_data, committee_table)


    # From MongoDB: Create list of unique bill_id & session for API call for bill details 
    unique_bill_session = bill_table.aggregate(
        { '$group': {
                '_id': {'bill_id': '$bill_id', 'session': '$session'}
        }})

    a = {}
    for k, val in unique_bill_session.iteritems():
        a = unique_bill_session[k]

    list_bill_session = []

    for i,entry in enumerate(a):
        x = ((a[i]['_id']['bill_id'],a[i]['_id']['session']))
        list_bill_session.append(x)
        
    #API calls for bill details using unique list of bill_id & session
    #example: sunlight.openstates.bill_detail(state="CA",session='20092010', bill_id='SCR 2')

    for i,entry in enumerate(list_bill_session):
        s = str(list_bill_session[i][1])
        b = str(list_bill_session[i][0])
        bill_details = sunlight.openstates.bill_detail(state='CA',session=s, bill_id=b)
        bill_d_table.insert(bill_details)


if __name__ == '__main__':
    main()

print("\n=======================Sucess!=====================================")




                
