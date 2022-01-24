from sqlalchemy import create_engine, Column, Integer, String, MetaData, ARRAY, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DB_LOCATION = '/mnt/raid1/home/bar_cohen/Shoham_KG.db'
Base = declarative_base()


class Crop(Base):
    __tablename__ = 'shoham_kg'
    label = Column(String)
    im_name = Column(String, primary_key=True)  # unique name for every crop
    frame_num = Column(Integer)
    bbox = ARRAY(Integer)
    vid_name = Column(String)
    track_id = Column(Integer)
    cam_id = Column(Integer)
    reviewed = Column(Boolean)
    crop_id = Column(Integer)
    is_face = Column(Boolean)
    is_vague = Column(Boolean)

    def set_im_name(self):
        self.im_name = f'v{self.vid_name}_f{self.frame_num}_b{str(list(self.bbox))}.png'


def create_session(db_location: str = DB_LOCATION):
    engine = create_engine(f'sqlite:///{db_location}', echo=True)  # should include the path to the db file
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


def get_entries():
    """
    Return all entries from the database according to the given filters. If no filters are given, return all entries.
    The returned result is a list of DbCrops objects.
    """
    session = create_session()
    crops = session.query(Crop).filter(Crop.frame > 315000)
    return crops.all()


# if __name__ == '__main__':
#     create_table()
#     crop = '0001_c1_f0307006.jpg'
#     parts = crop.split('_')
#     crop1 = DbCrops(person_id=parts[0], im_name=crop, cam_id=int(parts[1][1:]), frame=int(parts[2][1:-4]), x=1, y=2,
#                     h=3, w=4, video='daniel', track_id=5, date=123456)
#     add_entry(crop1)

