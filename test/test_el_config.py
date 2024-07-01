from lm_service.linking.onto import EntityLinkerManager


def test_elconf(el_conf):
    conf = EntityLinkerManager.from_dict(el_conf)
    assert len(conf.linkers) == 2


def test_lcong(lconf):
    assert lconf.url is not None
