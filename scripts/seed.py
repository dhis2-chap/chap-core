import sys

from sqlmodel import Session, select

from chap_core.database.database import SessionWrapper, engine
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.database.tables import DataSet
from chap_core.data import DataSet as _DataSet
from chap_core.datatypes import FullData

#dataset = ISIMIP_dengue_harmonized['vietnam']

if __name__ == '__main__':
    folder_path = sys.argv[1] if len(sys.argv) > 1 else '/home/knut/Data/ch_data/'
    with SessionWrapper() as session:
        dataset = _DataSet.from_csv('%snicaragua_weekly_data.csv' % folder_path, FullData)
        polygons = open('%snicaragua.json' % folder_path, 'r').read()
        session.add_dataset('nicaragua', dataset, polygons)
