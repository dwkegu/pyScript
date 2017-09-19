from gaft.components.individual import GAIndividual

class CustomIndividual(GAIndividual):
    def __init__(self, sData, pData, ranges, encoding='binary', eps=0.001):
        self.sData = sData
        self.pData = pData
        GAIndividual.__init__(self,ranges, encoding, eps)

    def _init_variants(self):
        length = len(self.ranges)
        res = []
        for i in range(0, length):
            res.append(1)
            flightRow = self.sData[i]
            planeName = str(flightRow[6]).strip().lower()
            i = 0
            for row in self.pData:
                if str(row[0]).strip().lower()== planeName:
                    res.append(i)
                i += 1
        return res

