import os

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, MetaData, and_, or_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# DB_LOCATION = '/mnt/raid1/home/bar_cohen/Shoham_KG.db' ## NEVER CHANGE THIS !!!
DB_LOCATION = '/mnt/raid1/home/bar_cohen/42Street.db' ## NEVER CHANGE THIS !!!
SAME_DAY_DB_LOCATION = '/mnt/raid1/home/bar_cohen/42StreetSameDayDB_newer.db'
Base = declarative_base()


class Crop(Base):
    __tablename__ = '42Street'
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
        self.im_name = f'v_{self.vid_name}_f{self.frame_num}_bbox_{self.x1}_{self.y1}_{self.x2}_{self.y2}.png'


class SameDayCropV2(Base):
    __tablename__ = '42StreetSameDayDBV2'
    gt_label = Column(String)
    label = Column(String)
    part = Column(Integer)
    im_name = Column(String, primary_key=True)  # unique name for every crop
    face_im_name = Column(String)  # unique name for every crop
    frame_num = Column(Integer)
    x1_crop = Column(Integer)
    y1_crop = Column(Integer)
    x2_crop = Column(Integer)
    y2_crop = Column(Integer)
    x_face = Column(Integer)
    y_face = Column(Integer)
    w_face = Column(Integer)
    h_face = Column(Integer)
    face_conf = Column(Float)
    face_cos_sim = Column(Float)
    face_ranks_diff = Column(Float)
    vid_name = Column(String)
    track_id = Column(Integer)
    cam_id = Column(Integer)
    crop_id = Column(Integer)

    def set_im_name(self):
        self.im_name = f'v_{self.vid_name}_f{self.frame_num}_bbox_{self.x1_crop}_{self.y1_crop}_{self.x2_crop}_{self.y2_crop}.png'
        self.face_im_name = f'v_{self.vid_name}_f{self.frame_num}_bbox_{self.x_face}_{self.y_face}_{self.w_face}_{self.h_face}.png'


def create_session(db_location: str = DB_LOCATION):
    engine = create_engine(f'sqlite:///{db_location}', echo=False)  # should include the path to the db file
    Session = sessionmaker(bind=engine, autoflush=False)
    return Session()


def create_table(db_location: str = DB_LOCATION):
    """
    Creates a new table in the database with the given location. Will create a new table with the name defined in the
    global class if it doesn't exist already.
    """
    engine = create_engine(f'sqlite:///{db_location}', echo=False)  # should include the path to the db file
    Base.metadata.create_all(engine)


def add_entries(crops: list, db_location: str = DB_LOCATION):
    """
    Adds the given DbCrop object to the database
    """
    session = create_session(db_location)
    session.add_all(crops)
    session.commit()


def delete_entries(delete_filter, db_location: str = DB_LOCATION):
    """ Usage example: delete_entries(delete_filter=Crop.vid_name == 'part1_s16000_e16501') """
    session = create_session(db_location)
    delete_q = Crop.__table__.delete().where(delete_filter)
    session.execute(delete_q)
    session.commit()


def get_entries(session=None, filters: tuple = None, op: str = 'AND', order=None, group=None, distinct_by=None,
                db_path=DB_LOCATION, crop_type=Crop):
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
    sql_query = session.query(crop_type).filter(query)
    if order:
        sql_query = sql_query.order_by(order)
    if group:
        sql_query = sql_query.group_by(group)
    if distinct_by:
        sql_query = sql_query.distinct(distinct_by)
    return sql_query


if __name__ == '__main__':
    create_table(db_location=SAME_DAY_DB_LOCATION)

