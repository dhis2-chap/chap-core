class ProgramConfig:
    def __init__(self, dhis2Username, dhis2Password, dhis2Baseurl):
        self.dhis2Username = dhis2Username
        self.dhis2Password = dhis2Password
        self.dhis2Baseurl = dhis2Baseurl

class DHIS2PullConfig:
    def __init__(self, dataElementId, periode, organisationUnit):
        self.dataElementId = dataElementId
        self.periode = periode
        self.organisationUnit = organisationUnit 