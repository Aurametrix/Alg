# a person generator
define:
  min_age: 10
  minor_age: 13
  working_age: 18

fields:
  age:
    random: gauss(25, 5)
    # minimum age is $min_age
    finalize: max($min_age, value)

  gender:
    mixture:
      - value: M
      - value: F

  name: "#{name.name}"
  job:
    value: "#{job.title}"
    onlyif: this.age > $working_age

  address:
    template: address/usa.yaml
  phone: # add a phone if the person is older than the minor age
    template: device/phone.yaml
    onlyif: this.age > ${minor_age}

  # we model our height as a gaussian that varies based on
  # age and gender
  height:
    lambda: this._base_height * this._age_factor
  _base_height:
    switch:
      - onlyif: this.gender == "F"
        random: gauss(60, 5)
      - onlyif: this.gender == "M"
        random: gauss(70, 5)

  _age_factor:
    switch:
      - onlyif: this.age < 15
        lambda: 1 - (20 - (this.age + 5)) / 20
      - default:
        value: 1
