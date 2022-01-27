import os

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, MetaData, and_, or_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DB_LOCATION = '/mnt/raid1/home/bar_cohen/Shoham_KG.db'
Base = declarative_base()


class Crop(Base):
    __tablename__ = 'shoham_kg'
    label = Column(String)
    im_name = Column(String, primary_key=True)  # unique name for every crop
    frame_num = Column(Integer)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    conf = Column(Float)
    vid_name = Column(String)
    track_id = Column(Integer)
    cam_id = Column(Integer)
    reviewed_one = Column(Boolean)
    reviewed_two = Column(Boolean)
    crop_id = Column(Integer)
    is_face = Column(Boolean)
    is_vague = Column(Boolean)
    invalid = Column(Boolean)

    def set_im_name(self):
        self.im_name = f'v{self.vid_name}_f{self.frame_num}_bbox_{self.x1}_{self.y1}_{self.x2}_{self.y2}.png'


def create_session(db_location: str = DB_LOCATION):
    engine = create_engine(f'sqlite:///{db_location}', echo=False)  # should include the path to the db file
    Session = sessionmaker(bind=engine)
    return Session()


def create_table(db_location: str = DB_LOCATION):
    """
    Creates a new table in the database with the given location. Will create a new table with the name defined in the
    global class if it doesn't exist already.
    """
    engine = create_engine(f'sqlite:///{db_location}', echo=True)  # should include the path to the db file
    Base.metadata.create_all(engine)


def add_entries(crops: list, db_location: str = DB_LOCATION):
    """
    Adds the given DbCrop object to the database
    """
    session = create_session(db_location)
    session.add_all(crops)
    session.commit()


def get_entries(session=None, filters: tuple = None, op: str = 'AND', order=None, group=None, distinct_by=None,
                db_path=DB_LOCATION):
    """
    Return all entries from the database according to the given filters. If no filters are given, return all entries.
    Args:
        - filters: a tuple of filters that should be used for querying the DB, for example (Crop.label == 'Daniel').
        - op: the operator that should be used between the given filters, options: 'AND' / 'OR'.
        - order: the column in the DB according to which the returned results should be ordered.
        - group: the column by which the returned results should be grouped.
    Return:
        The returned result is a list of Crops objects matching the given filters.
    """
    if not session:
        session = create_session(db_path)
    if op == 'AND':
        query = and_(*filters)
    elif op == 'OR':
        query = or_(*filters)
    else:
        raise Exception('Invalid query operator, valid options are: AND, OR')
    sql_query = session.query(Crop).filter(query)
    if order:
        sql_query = sql_query.order_by(order)
    if group:
        sql_query = sql_query.group_by(group)
    if distinct_by:
        sql_query = sql_query.distinct(distinct_by)
    return sql_query


# if __name__ == '__main__':
#     create_table()
    # vid_name = '1.8.21-095724'
    # crops = get_entries(filters=({Crop.vid_name == vid_name}), db_path=DB_LOCATION_ORIG)
#     crop = '0001_c1_f0307006.jpg'
#     parts = crop.split('_')
#     crop1 = DbCrops(person_id=parts[0], im_name=crop, cam_id=int(parts[1][1:]), frame=int(parts[2][1:-4]), x=1, y=2,
#                     h=3, w=4, video='daniel', track_id=5, date=123456)
#     add_entry(crop1)

