The [cBioPortal for Cancer Genomics](http://www.cbioportal.org) provides visualization, analysis and download of large-scale cancer genomics data sets. 
cBioPortal provides a [REST API for programmatic access to the data](https://docs.cbioportal.org/web-api-and-clients/)

[Datahub](https://github.com/cBioPortal/datahub)

[cBio Portal Documentation](http://www.cbioportal.org/tutorials)
[Code Repository](https://github.com/cBioPortal/)

[Tutorial on cBioPortal's REST API](https://docs.google.com/presentation/d/12SI8crS3oXvnm0YRIm_IDSvoSjg4ZKXYm-kezahCXRM/edit#slide=id.g8082cf6af9_0_220)

Bravado/ Swagger - a library for using Swagger defined API's.

    from bravado.client import SwaggerClient
    cbioportal = SwaggerClient.from_url('https://www.cbioportal.org/api/api-docs',
                                    config={"validate_requests":False,"validate_responses":False,"validate_swagger_spec": False})
Endpoints:
>>> dir(cbioportal)
['Cancer Types',
 'Clinical Attributes',
 'Clinical Data',
 'Clinical Events',
 'Copy Number Segments',
 'Discrete Copy Number Alterations',
 'Gene Panels',
 'Genes',
 'Molecular Data',
 'Molecular Profiles',
 'Mutations',
 'Patients',
 'Sample Lists',
 'Samples',
 'Studies']
 
 cbio_py client provides a simple wrapper for the API:  https://pypi.org/project/cbio-py/
 
 https://www.cbioportal.org/api/swagger-ui.html#/Samples
 
 connect to the cBioPortal API:
 
     from bravado.client import SwaggerClient
     cbioportal = SwaggerClient.from_url('https://www.cbioportal.org/api/api-docs',
                                config={"validate_requests":False,"validate_responses":False})
     print(cbioportal)

acc = cbioportal.Cancer_Types.getCancerTypeUsingGET(cancerTypeId='acc').result()
print(acc)
