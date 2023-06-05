Array<T> *V = NULL;
    if (nonmax == 1) {
        dim4 V_dims(in_dims[0], in_dims[1]);
Expand Down
Expand Up
	@@ -290,6 +289,8 @@ features fast(const Array<T> &in, const float thr, const unsigned arc_length,
                       *x_nonmax, *y_nonmax, *score_nonmax,
                       &count, feat_found);

        feat_found = std::min(max_feat, count);
        feat_found_dims = dim4(feat_found);

Expand All
	@@ -315,6 +316,9 @@ features fast(const Array<T> &in, const float thr, const unsigned arc_length,
        delete x;
        delete y;
        delete score;
    }
    else {
        // TODO: improve output copy, use af_index?
Expand All
	@@ -335,6 +339,10 @@ features fast(const Array<T> &in, const float thr, const unsigned arc_length,
            y_out_ptr[i] = y_ptr[i];
            score_out_ptr[i] = score_ptr[i];
        }
    }

    features feat;
Expand All
	@@ -345,6 +353,13 @@ features fast(const Array<T> &in, const float thr, const unsigned arc_length,
    feat.setOrientation(getHandle<float>(*orientation_out));
    feat.setSize(getHandle<float>(*size_out));

    return feat;
}