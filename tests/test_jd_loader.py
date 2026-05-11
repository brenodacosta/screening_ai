from api.jd_loader import load_jd


def test_jd_parses(jd_path):
    jd = load_jd(jd_path)
    assert jd.title == "Delivery Driver"
    assert "spain" in jd.service_areas and "mexico" in jd.service_areas


def test_service_areas_count(jd_path):
    jd = load_jd(jd_path)
    assert len(jd.all_service_areas()) == 45


def test_service_area_match_is_accent_insensitive(jd_path):
    jd = load_jd(jd_path)
    # "ciudad de mexico" (no accent, lowercase) should match "Ciudad de México"
    assert jd.is_in_service_area("ciudad de mexico") is True
    assert jd.is_in_service_area("MÁLAGA") is True
    assert jd.is_in_service_area("Buenos Aires") is False


def test_required_fields_present(jd_path):
    jd = load_jd(jd_path)
    assert jd.employment_types and jd.shifts
    assert "es" in jd.languages_supported and "en" in jd.languages_supported
