class TimeUtil:

    @staticmethod
    def get_m_s(time):
        minute = time // 60
        second = time % 60
        return minute, second
