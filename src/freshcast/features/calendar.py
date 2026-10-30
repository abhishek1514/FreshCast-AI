"""Calendar and temporal feature engineering for foodservice demand."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


class CalendarFeatureExtractor:
    """Extracts calendar, cyclical, and foodservice ordering cycle features."""

    def __init__(self, country: str = "US"):
        self.country = country
        self.holiday_calendar = holidays.country_holidays(self.country)

    def transform(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        df = df.copy()
        dates = pd.to_datetime(df[date_col])

        # Basic calendar features
        df["day_of_week"] = dates.dt.day_of_week
        df["day_of_month"] = dates.dt.day
        df["month"] = dates.dt.month
        df["quarter"] = dates.dt.quarter
        df["day_of_year"] = dates.dt.day_of_year
        df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        df["year"] = dates.dt.year

        # Cyclical transforms (preserve circular continuity: Sunday is close to Monday)
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # Foodservice industry indicators
        # Monday (0) = Heavy restocking after weekend restaurant depletion
        df["is_monday_restock"] = (df["day_of_week"] == 0).astype(int)
        # Thursday/Friday (3, 4) = Peak restaurant prep for weekend dining rush
        df["is_weekend_prep"] = df["day_of_week"].isin([3, 4]).astype(int)
        # Weekend warehouse delivery (skeleton staffing)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Holiday features
        unique_dates = pd.Series(dates.unique())
        holiday_dict = {d: int(d in self.holiday_calendar) for d in unique_dates}
        df["is_holiday"] = dates.map(holiday_dict).fillna(0).astype(int)

        # Pre-holiday rush indicator (e.g. within 3 days before a holiday)
        holiday_dates = set(d for d in unique_dates if d in self.holiday_calendar)
        days_to_holiday = []
        for d in dates:
            upcoming = [h for h in holiday_dates if 0 <= (h - d).days <= 7]
            if upcoming:
                min_days = min((h - d).days for h in upcoming)
                days_to_holiday.append(min_days)
            else:
                days_to_holiday.append(99)

        df["days_to_holiday"] = days_to_holiday
        df["is_holiday_week"] = (df["days_to_holiday"] <= 5).astype(int)

        return df

