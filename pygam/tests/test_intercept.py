# -*- coding: utf-8 -*-

import pytest

from pygam import LinearGAM

class TestIntercept(object):
    def test_basic_intercept(self, mcycle_X_y):
        """
        basic intercept
        """
        X, y = mcycle_X_y
        gam = LinearGAM(fit_intercept=True).fit(X, y)
        pred = gam.predict(X)
        pdep = gam.partial_dependence(term=-1)

        assert pred.shape[0] == X.shape[0]
        assert pdep.shape[0] == 100

        X0 = gam.generate_X_grid(term=0, n=500, meshgrid=False)
        assert X0.shape[0] == 500

        with pytest.raises(ValueError):
            gam.generate_X_grid(term=1, n=500, meshgrid=False)

        X1 = gam.generate_X_grid(term=-1, n=500, meshgrid=True)
        assert X1[0].shape[0] == 500

    def test_only_intercept(self, mcycle_X_y):
        """
        basic intercept
        """
        X, y = mcycle_X_y
        gam = LinearGAM(terms=None, fit_intercept=True).fit(X, y)
        pred = gam.predict(X)

        with pytest.raises(ValueError):
            gam.partial_dependence(term=-1)

        assert pred.shape[0] == X.shape[0]

        with pytest.raises(ValueError):
            gam.generate_X_grid(term=0, n=500, meshgrid=False)
