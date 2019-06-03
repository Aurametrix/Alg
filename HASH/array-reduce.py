function keyByUsernameReducer(acc, person) {
    return {...acc, [person.username]: person};
}
const peopleObj = peopleArr.reduce(keyByUsernameReducer, {});
console.log(peopleObj);
// ⦘ {
//     "glestrade": {
//         "username":    "glestrade",
//         "displayname": "Inspector Lestrade",
//         "email":       "glestrade@met.police.uk",
//         "authHash":    "bdbf9920f42242defd9a7f76451f4f1d",
//          "lastSeen":    "2019-05-13T11:07:22+00:00"
//     },
//     "mholmes": {
//         "username":    "mholmes",
//         "displayname": "Mycroft Holmes",
//         "email":       "mholmes@gov.uk",
//         "authHash":    "b4d04ad5c4c6483cfea030ff4e7c70bc",
//          "lastSeen":    "2019-05-10T11:21:36+00:00"
//     },
//     "iadler":{
//         "username":    "iadler",
//         "displayname": "Irene Adler",
//         "email":       null,
//         "authHash":    "319d55944f13760af0a07bf24bd1de28",
//          "lastSeen":    "2019-05-17T11:12:12+00:00"
//     }
// }
