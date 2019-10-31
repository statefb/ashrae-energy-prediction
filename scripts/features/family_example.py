class FamilySize(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1

FamilySize().run().save()
