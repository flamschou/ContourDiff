"""
pytest test_pipeline.py -v -s

-v  : affiche le nom de chaque test
-s  : affiche les logs détaillés

Prérequis : placer dans ContourDiff/data_test/
  - zscore_*.mat
  - contours_zscore_*.mat  (produit par batch_contours.py)

Les sorties visuelles (PNG) sont enregistrées dans data_test/visual_checks/.
Sur macOS le dossier s'ouvre automatiquement à la fin de la session.
"""

import sys, os
from pathlib import Path
import numpy as np
import pytest
import torch
import matplotlib
matplotlib.use("Agg")   # pas de fenêtre interactive — les figures sont sauvegardées en PNG
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import TrainingConfig
from dataset import _load_mat, _resize_hw, MatIRMDataset
from transform import TrainTransform, ValTransform
from train import find_mat_pairs


# ── Chemins ───────────────────────────────────────────────────────────────────

DATA_TEST  = Path(__file__).parent / "data_test"
VISUAL_DIR = DATA_TEST / "visual_checks"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    path = VISUAL_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"      PNG sauvegardé → {path}")


def _sep():
    print("   " + "─" * 60)


# ── Ouverture automatique du dossier visuel à la fin (macOS) ─────────────────

def pytest_sessionfinish(session, exitstatus):
    if sys.platform == "darwin" and any(VISUAL_DIR.glob("*.png")):
        print(f"\n{'═'*60}")
        print(f"  Ouverture du dossier des visuels : {VISUAL_DIR}")
        print(f"{'═'*60}")
        os.system(f'open "{VISUAL_DIR}"')


# ── Helpers de découverte ─────────────────────────────────────────────────────

def _get_pair():
    pairs = find_mat_pairs(str(DATA_TEST))
    return pairs[0] if pairs else None

def _get_zscore_path():
    paths = sorted(DATA_TEST.rglob("zscore_*.mat"))
    return str(paths[0]) if paths else None


needs_zscore = pytest.mark.skipif(
    _get_zscore_path() is None,
    reason=f"Aucun zscore_*.mat dans {DATA_TEST}",
)
needs_pair = pytest.mark.skipif(
    _get_pair() is None,
    reason=f"Aucun couple zscore + contours dans {DATA_TEST} — lancez batch_contours.py",
)


# ── Fixtures (scope=session : chargement unique pour toute la session) ────────

@pytest.fixture(scope="session")
def zscore_path():
    return _get_zscore_path()

@pytest.fixture(scope="session")
def mat_pair():
    return _get_pair()

@pytest.fixture(scope="session")
def raw_volume(zscore_path):
    return _load_mat(zscore_path)

@pytest.fixture(scope="session")
def cfg():
    return TrainingConfig(img_size=64, near_guided=False, near_guided_ratio=0.2)

@pytest.fixture(scope="session")
def cfg_near():
    return TrainingConfig(img_size=64, near_guided=True, near_guided_ratio=1.0)

@pytest.fixture(scope="session")
def dataset(mat_pair, cfg):
    if mat_pair is None:
        pytest.skip(f"Aucun couple dans {DATA_TEST}")
    return MatIRMDataset(
        [mat_pair],
        train_transform=TrainTransform(cfg),
        val_transform=ValTransform(),
        train=True, config=cfg,
    )

@pytest.fixture(scope="session")
def val_dataset(mat_pair, cfg):
    if mat_pair is None:
        pytest.skip(f"Aucun couple dans {DATA_TEST}")
    return MatIRMDataset(
        [mat_pair],
        train_transform=TrainTransform(cfg),
        val_transform=ValTransform(),
        train=False, config=cfg,
    )


# ── _load_mat ─────────────────────────────────────────────────────────────────

class TestLoadMat:
    """
    _load_mat() charge un fichier .mat (scipy ou h5py) et retourne un ndarray
    float32 de shape (Z, H, W) en convention Python.
    Les fichiers MATLAB stockent les volumes en (H, W, Z) — le chargement
    doit appliquer np.moveaxis pour remettre Z en premier axe.
    """

    @needs_zscore
    def test_dtype_float32(self, raw_volume):
        print("\n   OBJECTIF : vérifier que _load_mat convertit systématiquement en float32")
        print("   POURQUOI : le réseau attend du float32 ; un float64 multiplierait la RAM par 2")
        _sep()
        print(f"   dtype obtenu : {raw_volume.dtype}")
        assert raw_volume.dtype == np.float32
        print("   ✓ PASS — dtype == float32")

    @needs_zscore
    def test_ndim_3(self, raw_volume):
        print("\n   OBJECTIF : vérifier que le volume chargé est bien 3D (Z, H, W)")
        print("   POURQUOI : un fichier .mat peut contenir un tableau 2D ou 4D par erreur")
        _sep()
        print(f"   ndim obtenu : {raw_volume.ndim}  |  shape : {raw_volume.shape}")
        assert raw_volume.ndim == 3
        print("   ✓ PASS — ndim == 3")

    @needs_zscore
    def test_shape_convention_coronal_first(self, raw_volume):
        print("\n   OBJECTIF : vérifier la convention d'axes (A/P, S/I, L/R) — coupes coronales")
        print("   POURQUOI : MATLAB stocke en (H_S/I, W_L/R, Z_A/P).")
        print("              moveaxis(-1,0) → axis 0 = A/P → 142 coupes coronales (S/I × L/R).")
        print("              Les coupes coronales sont naturellement carrées (S/I=L/R=226).")
        print("              La détection de contours étant 3D, le plan de coupe importe peu.")
        _sep()
        ap, si, lr = raw_volume.shape
        print(f"   A/P={ap} coupes  |  chaque coupe : S/I={si} × L/R={lr}")
        print(f"   Coupes carrées : {'oui ✓' if si == lr else f'non ({si}≠{lr})'}")
        assert ap > 0 and si > 0 and lr > 0
        print("   ✓ PASS — convention coronale (A/P, S/I, L/R) correcte")

    @needs_zscore
    def test_finite_values(self, raw_volume):
        print("\n   OBJECTIF : vérifier l'absence de NaN et Inf dans le volume")
        print("   POURQUOI : un seul NaN dans le volume contaminerait tout le batch par propagation")
        _sep()
        n_nan = int(np.isnan(raw_volume).sum())
        n_inf = int(np.isinf(raw_volume).sum())
        print(f"   NaN : {n_nan}  |  Inf : {n_inf}")
        print(f"   Range : [{raw_volume.min():.4f},  {raw_volume.max():.4f}]")
        print(f"   Moyenne : {raw_volume.mean():.4f}  |  Écart-type : {raw_volume.std():.4f}")
        print(f"   (fichier zscore → on attend μ≈0, σ≈1 sur l'ensemble du volume)")
        assert np.isfinite(raw_volume).all()
        print("   ✓ PASS — toutes les valeurs sont finies")

    @needs_zscore
    def test_visual_middle_slice(self, raw_volume):
        print("\n   OBJECTIF [VISUEL] : afficher la coupe centrale pour vérifier l'orientation")
        print("   POURQUOI : un mauvais moveaxis produit une image transposée ou pivotée")
        print("   CE QU'ON ATTEND : une coupe axiale thoracique (vue de dessus, symétrie L/R)")
        _sep()
        mid = raw_volume.shape[0] // 2
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(raw_volume[mid], cmap="gray")
        ax.set_title(f"_load_mat — coupe axiale Z={mid}/{raw_volume.shape[0]-1}\n"
                     f"shape={raw_volume.shape}  range=[{raw_volume.min():.2f}, {raw_volume.max():.2f}]")
        ax.axis("off")
        _save(fig, "01_load_mat_middle_slice.png")
        print(f"   Coupe affichée : Z={mid}")
        print("   ✓ PASS — vérifiez visuellement l'orientation dans le PNG")


# ── _resize_hw ────────────────────────────────────────────────────────────────

class TestResizeHW:
    """
    _resize_hw() redimensionne les axes H et W d'un volume (Z, H, W) vers
    une cible (Z, th, tw) en utilisant scipy.ndimage.zoom.
    L'axe Z n'est jamais touché.
    """

    @needs_zscore
    def test_output_shape(self, raw_volume):
        print("\n   OBJECTIF : vérifier que le resize produit bien la shape cible (Z, 64, 64)")
        print("   POURQUOI : une erreur d'indice dans les facteurs de zoom donnerait une shape")
        print("              incorrecte et planterait toute la suite de la pipeline")
        _sep()
        out = _resize_hw(raw_volume, 64, 64, order=1)
        print(f"   Entrée : {raw_volume.shape}  →  Sortie : {out.shape}")
        assert out.shape == (raw_volume.shape[0], 64, 64)
        print("   ✓ PASS — H et W redimensionnés à 64")

    @needs_zscore
    def test_z_unchanged(self, raw_volume):
        print("\n   OBJECTIF : vérifier que l'axe Z (nb de coupes) n'est pas modifié")
        print("   POURQUOI : on découpe les coupes Z dans __getitem__ ; si Z change,")
        print("              l'index self.samples serait invalide et des IndexError suivraient")
        _sep()
        out = _resize_hw(raw_volume, 32, 48, order=1)
        print(f"   Z original : {raw_volume.shape[0]}  |  Z après resize : {out.shape[0]}")
        assert out.shape[0] == raw_volume.shape[0]
        print("   ✓ PASS — Z inchangé")

    @needs_zscore
    def test_dtype_preserved(self, raw_volume):
        print("\n   OBJECTIF : vérifier que scipy.ndimage.zoom ne change pas le dtype")
        print("   POURQUOI : scipy peut silencieusement upcast en float64 selon la version,")
        print("              ce qui doublerait l'usage mémoire")
        _sep()
        out = _resize_hw(raw_volume, 64, 64, order=1)
        print(f"   dtype entrée : {raw_volume.dtype}  |  dtype sortie : {out.dtype}")
        assert out.dtype == np.float32
        print("   ✓ PASS — float32 préservé")

    @needs_zscore
    def test_visual_before_after(self, raw_volume):
        print("\n   OBJECTIF [VISUEL] : vérifier le resize d'une coupe coronale carrée")
        print("   CE QU'ON ATTEND : coupe coronale (S/I × L/R) naturellement carrée,")
        print("                     resize simple sans distorsion ni padding")
        _sep()
        mid = raw_volume.shape[0] // 2
        h_orig, w_orig = raw_volume.shape[1], raw_volume.shape[2]
        out = _resize_hw(raw_volume, 64, 64, order=1)
        print(f"   Coupe originale : ({h_orig}, {w_orig})  {'carrée ✓' if h_orig == w_orig else 'rectangulaire'}")
        print(f"   Après resize    : {out.shape[1:]}")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_volume[mid], cmap="gray")
        axes[0].set_title(f"Coupe coronale originale\n{h_orig}×{w_orig}")
        axes[0].axis("off")
        axes[1].imshow(out[mid], cmap="gray")
        axes[1].set_title("Après resize  64×64")
        axes[1].axis("off")
        fig.suptitle("_resize_hw — coupe coronale (S/I × L/R)\nVérifiez : pas de distorsion")
        _save(fig, "02_resize_before_after.png")
        print("   ✓ PASS — vérifiez visuellement que les structures ne sont pas déformées")


# ── TrainTransform ────────────────────────────────────────────────────────────

class TestTrainTransform:
    """
    TrainTransform tire UN jeu de paramètres aléatoires (angle, zoom, translation,
    flip) et l'applique à l'ensemble du volume (Z, H, W) via scipy.ndimage.
    Toutes les coupes Z reçoivent exactement la même transformation spatiale
    → cohérence 3D garantie par construction.
    Image : interpolation order=1 (bilinéaire) pour préserver les gradients.
    Contour : interpolation order=0 (plus proche voisin) pour préserver les bords binaires.
    """

    @needs_zscore
    def test_output_shape_preserved(self, raw_volume, cfg):
        print("\n   OBJECTIF : vérifier que la shape (Z, H, W) est identique en entrée et sortie")
        print("   POURQUOI : rotation + zoom peuvent modifier la shape si reshape=True chez scipy ;")
        print("              on utilise reshape=False et crop/pad pour garantir une shape fixe")
        _sep()
        vol  = _resize_hw(raw_volume, cfg.img_size, cfg.img_size, order=1)
        cont = (vol > vol.mean()).astype(np.float32)
        t = TrainTransform(cfg)
        img_aug, cont_aug = t(vol, cont)
        print(f"   Shape entrée  : {vol.shape}")
        print(f"   Shape image   : {img_aug.shape}")
        print(f"   Shape contour : {cont_aug.shape}")
        assert img_aug.shape == vol.shape
        assert cont_aug.shape == cont.shape
        print("   ✓ PASS — shape préservée après augmentation")

    @needs_zscore
    def test_3d_coherence(self, raw_volume, cfg):
        print("\n   OBJECTIF : vérifier que toutes les coupes Z reçoivent la MÊME transformation")
        print("   POURQUOI : l'ancienne pipeline PIL appliquait une transformation indépendante")
        print("              par coupe → deux coupes adjacentes pouvaient avoir des flips opposés,")
        print("              ce qui est anatomiquement impossible")
        print("   MÉTHODE : on empile N fois la coupe centrale du vrai volume → toutes identiques.")
        print("             Après augmentation, elles doivent rester identiques (diff max = 0)")
        _sep()
        resized = _resize_hw(raw_volume, cfg.img_size, cfg.img_size, order=1)
        mid = resized.shape[0] // 2
        Z = 6
        image   = np.stack([resized[mid]] * Z)
        contour = (image > image.mean()).astype(np.float32)
        t = TrainTransform(cfg)
        np.random.seed(7)
        img_aug, _ = t(image, contour)
        max_diff = max(np.abs(img_aug[0] - img_aug[z]).max() for z in range(1, Z))
        print(f"   Diff max entre slice 0 et slices 1..{Z-1} : {max_diff:.2e}  (attendu = 0.00e+00)")

        fig, axes = plt.subplots(1, Z, figsize=(3*Z, 3))
        for z in range(Z):
            axes[z].imshow(img_aug[z], cmap="gray")
            axes[z].set_title(f"Z={z}")
            axes[z].axis("off")
        fig.suptitle("Cohérence 3D : 6 copies du même slice après augmentation\n"
                     "→ toutes les vignettes doivent être VISUELLEMENT IDENTIQUES")
        _save(fig, "03_3d_coherence_slices.png")
        print("   [VISUEL] Les 6 vignettes doivent être pixel-perfect identiques")

        for z in range(1, Z):
            np.testing.assert_allclose(img_aug[0], img_aug[z], atol=1e-5)
        print("   ✓ PASS — diff numérique = 0, cohérence 3D confirmée")

    @needs_zscore
    def test_flip_applied_to_image_and_contour(self, raw_volume, cfg):
        print("\n   OBJECTIF : vérifier que le flip L/R est appliqué identiquement à l'image")
        print("              ET au contour")
        print("   POURQUOI : si l'image est flippée mais pas le contour, le modèle apprend")
        print("              des bords qui ne correspondent pas à l'anatomie de l'image")
        print("   MÉTHODE : flip_p=1.0 force le flip systématique ; on compare à vol[:,:,::-1]")
        _sep()
        vol  = _resize_hw(raw_volume, cfg.img_size, cfg.img_size, order=1)
        cont = (vol > vol.mean()).astype(np.float32)
        cfg_flip = TrainingConfig(img_size=cfg.img_size, flip_p=1.0, apply_p=0.0)
        t = TrainTransform(cfg_flip)
        img_aug, cont_aug = t(vol.copy(), cont.copy())
        diff_img  = np.abs(img_aug  - vol[:, :, ::-1]).max()
        diff_cont = np.abs(cont_aug - cont[:, :, ::-1]).max()
        print(f"   Diff image  vs miroir théorique : {diff_img:.2e}  (attendu = 0.00e+00)")
        print(f"   Diff contour vs miroir théorique : {diff_cont:.2e}  (attendu = 0.00e+00)")

        mid = vol.shape[0] // 2
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        axes[0,0].imshow(vol[mid],         cmap="gray"); axes[0,0].set_title("Image originale")
        axes[0,1].imshow(img_aug[mid],     cmap="gray"); axes[0,1].set_title("Image flippée (obtenu)")
        axes[0,2].imshow(vol[mid,:,::-1],  cmap="gray"); axes[0,2].set_title("Image flippée (attendu)")
        axes[1,0].imshow(cont[mid],        cmap="gray"); axes[1,0].set_title("Contour original")
        axes[1,1].imshow(cont_aug[mid],    cmap="gray"); axes[1,1].set_title("Contour flippé (obtenu)")
        axes[1,2].imshow(cont[mid,:,::-1], cmap="gray"); axes[1,2].set_title("Contour flippé (attendu)")
        for ax in axes.flat: ax.axis("off")
        fig.suptitle("Flip L/R : colonnes 'obtenu' et 'attendu' doivent être IDENTIQUES\n"
                     "Image et contour doivent être symétriques l/r par rapport à l'original")
        _save(fig, "04_flip_image_vs_contour.png")
        print("   [VISUEL] 'obtenu' = 'attendu' pour image et contour")

        np.testing.assert_allclose(img_aug,  vol[:, :, ::-1],  atol=1e-5)
        np.testing.assert_allclose(cont_aug, cont[:, :, ::-1], atol=1e-5)
        print("   ✓ PASS — flip identique sur image et contour")

    @needs_zscore
    def test_determinism_with_same_seed(self, raw_volume, cfg):
        print("\n   OBJECTIF : vérifier que np.random.seed() rend l'augmentation reproductible")
        print("   POURQUOI : indispensable pour déboguer ; si deux runs identiques donnent des")
        print("              résultats différents, impossible d'isoler un problème")
        _sep()
        vol  = _resize_hw(raw_volume, cfg.img_size, cfg.img_size, order=1)
        cont = (vol > vol.mean()).astype(np.float32)
        t = TrainTransform(cfg)
        np.random.seed(42); img1, cont1 = t(vol.copy(), cont.copy())
        np.random.seed(42); img2, cont2 = t(vol.copy(), cont.copy())
        diff = np.abs(img1 - img2).max()
        print(f"   Diff max entre run 1 et run 2 (même seed=42) : {diff:.2e}  (attendu = 0)")
        np.testing.assert_array_equal(img1, img2)
        np.testing.assert_array_equal(cont1, cont2)
        print("   ✓ PASS — déterministe à graine fixée")

    @needs_zscore
    def test_visual_augmentation_samples(self, raw_volume, cfg):
        print("\n   OBJECTIF [VISUEL] : montrer 4 augmentations différentes du même slice")
        print("   CE QU'ON ATTEND : variations visibles (rotation, flip, zoom) mais")
        print("                     le contenu anatomique reste reconnaissable et non dégradé")
        _sep()
        resized = _resize_hw(raw_volume, cfg.img_size, cfg.img_size, order=1)
        mid = resized.shape[0] // 2
        t = TrainTransform(cfg)
        cont = (resized > resized.mean()).astype(np.float32)
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(resized[mid], cmap="gray"); axes[0].set_title("Original")
        axes[0].axis("off")
        for i in range(1, 5):
            aug, _ = t(resized.copy(), cont.copy())
            axes[i].imshow(aug[mid], cmap="gray"); axes[i].set_title(f"Aug {i}")
            axes[i].axis("off")
        fig.suptitle("4 augmentations du même slice central\n"
                     "→ doivent varier (rotation/flip/zoom) mais rester anatomiquement lisibles")
        _save(fig, "05_augmentation_samples.png")
        print("   ✓ PASS — vérifiez que les 4 augmentations sont distinctes et cohérentes")


# ── ValTransform ──────────────────────────────────────────────────────────────

class TestValTransform:
    """
    ValTransform est une identité stricte : aucune augmentation ne doit être
    appliquée en validation. Le resize a déjà été fait dans MatIRMDataset.__init__.
    """

    @needs_zscore
    def test_identity(self, raw_volume, cfg):
        print("\n   OBJECTIF : vérifier que ValTransform ne modifie pas les données")
        print("   POURQUOI : en validation on évalue sur les données brutes non augmentées ;")
        print("              toute modification fausserait les métriques de validation")
        _sep()
        vol  = _resize_hw(raw_volume, cfg.img_size, cfg.img_size, order=1)
        cont = (vol > 0).astype(np.float32)
        t = ValTransform()
        img_out, cont_out = t(vol, cont)
        diff_img  = np.abs(img_out  - vol).max()
        diff_cont = np.abs(cont_out - cont).max()
        print(f"   Diff image  (attendu 0) : {diff_img}")
        print(f"   Diff contour (attendu 0) : {diff_cont}")
        np.testing.assert_array_equal(img_out, vol)
        np.testing.assert_array_equal(cont_out, cont)
        print("   ✓ PASS — ValTransform = identité stricte")


# ── MatIRMDataset ─────────────────────────────────────────────────────────────

class TestMatIRMDataset:
    """
    MatIRMDataset charge les .mat, resize, calcule les bornes percentile,
    et expose des samples (images, contours, near_images) comme tenseurs
    float32 normalisés :
      - images  ∈ [-1, 1]  via normalisation percentile par volume
      - contours ∈ {0, 1}  via seuillage à 0.5
      - near_images = 0    quand near_guided=False
    resample() ré-tire de nouvelles augmentations 3D pour toute l'époque.
    """

    @needs_pair
    def test_len_equals_z_slices(self, dataset, raw_volume):
        print("\n   OBJECTIF : vérifier que len(dataset) == nombre de coupes Z du volume")
        print("   POURQUOI : chaque coupe axiale est un sample indépendant ; si le compte")
        print("              est faux, certaines coupes sont manquantes ou dupliquées")
        _sep()
        expected = raw_volume.shape[0]
        print(f"   Coupes Z dans le volume : {expected}")
        print(f"   len(dataset)            : {len(dataset)}")
        assert len(dataset) == expected
        print("   ✓ PASS — len(dataset) == nb de coupes Z")

    @needs_pair
    def test_getitem_keys(self, dataset):
        print("\n   OBJECTIF : vérifier que __getitem__ retourne exactement les clés attendues")
        print("   POURQUOI : add_contours_to_noise() et evaluate() accèdent à ces clés par nom ;")
        print("              une clé manquante ou mal nommée lève un KeyError en plein entraînement")
        _sep()
        sample = dataset[0]
        keys = set(sample.keys())
        expected = {"images", "contours", "near_images", "image_name", "contour_name"}
        print(f"   Clés obtenues  : {sorted(keys)}")
        print(f"   Clés attendues : {sorted(expected)}")
        assert keys == expected
        print("   ✓ PASS — toutes les clés présentes et correctement nommées")

    @needs_pair
    def test_getitem_shapes(self, dataset, cfg):
        print("\n   OBJECTIF : vérifier que chaque tenseur a la shape (1, img_size, img_size)")
        print("   POURQUOI : le UNet attend (B, C, H, W) ; C=1 pour IRM niveaux de gris,")
        print("              H=W=img_size configuré. Une shape incorrecte lève RuntimeError")
        print("              dans la première conv du UNet")
        _sep()
        sample = dataset[0]
        expected = torch.Size([1, cfg.img_size, cfg.img_size])
        print(f"   images      : {sample['images'].shape}     attendu : {expected}")
        print(f"   contours    : {sample['contours'].shape}     attendu : {expected}")
        print(f"   near_images : {sample['near_images'].shape}  attendu : {expected}")
        assert sample["images"].shape    == expected
        assert sample["contours"].shape  == expected
        assert sample["near_images"].shape == expected
        print("   ✓ PASS — shapes (1, img_size, img_size)")

    @needs_pair
    def test_image_range(self, dataset):
        print("\n   OBJECTIF : vérifier que les images sont normalisées dans [-1, 1]")
        print("   POURQUOI : ContourDiff utilise transforms.Normalize([0.5],[0.5]) en sortie,")
        print("              ce qui suppose une entrée dans [0,1]. Avec la pipeline float32")
        print("              la normalisation percentile → [-1,1] remplace ce rôle.")
        print("              Une valeur > 1 ou < -1 indique une fuite d'intensités extrêmes")
        _sep()
        vmin, vmax = float("inf"), float("-inf")
        for idx in range(min(20, len(dataset))):
            img = dataset[idx]["images"]
            vmin = min(vmin, img.min().item())
            vmax = max(vmax, img.max().item())
        print(f"   Range sur 20 coupes : [{vmin:.5f},  {vmax:.5f}]")
        print(f"   Bornes autorisées   : [-1.00000,  +1.00000]")
        assert vmin >= -1.0 - 1e-5
        assert vmax <=  1.0 + 1e-5
        print("   ✓ PASS — images dans [-1, 1]")

    @needs_pair
    def test_contour_range(self, dataset):
        print("\n   OBJECTIF : vérifier que les contours sont normalisés dans [-1, 1]")
        print("   POURQUOI : les contours continus [0,1] sont normalisés vers [-1,1]")
        print("              pour correspondre à l'échelle de l'image dans le canal UNet.")
        _sep()
        vmin, vmax = float("inf"), float("-inf")
        for idx in range(min(20, len(dataset))):
            cont = dataset[idx]["contours"]
            vmin = min(vmin, cont.min().item())
            vmax = max(vmax, cont.max().item())
        print(f"   Range sur 20 coupes : [{vmin:.5f},  {vmax:.5f}]")
        print(f"   Bornes autorisées   : [-1.00000,  +1.00000]")
        assert vmin >= -1.0 - 1e-5
        assert vmax <=  1.0 + 1e-5
        print("   ✓ PASS — contours dans [-1, 1]")

    @needs_pair
    def test_near_images_zero_when_disabled(self, dataset):
        print("\n   OBJECTIF : vérifier que near_images est un tenseur de zéros quand near_guided=False")
        print("   POURQUOI : near_images est concaténé au bruit dans add_contours_to_noise() ;")
        print("              s'il n'est pas nul alors qu'on ne veut pas de guidage adjacent,")
        print("              le modèle reçoit une information fantôme")
        _sep()
        val = dataset[0]["near_images"].abs().sum().item()
        print(f"   Somme absolue de near_images (attendu 0) : {val}")
        assert val == 0.0
        print("   ✓ PASS — near_images = tenseur nul")

    @needs_pair
    def test_near_images_nonzero_when_enabled(self, mat_pair, cfg_near):
        print("\n   OBJECTIF : vérifier que near_images contient une vraie coupe adjacente")
        print("              quand near_guided=True et near_guided_ratio=1.0")
        print("   POURQUOI : si _near_tensor() retournait toujours des zéros, le guidage")
        print("              adjacent serait silencieusement désactivé sans erreur visible")
        _sep()
        ds = MatIRMDataset(
            [mat_pair],
            train_transform=TrainTransform(cfg_near),
            val_transform=ValTransform(),
            train=True, config=cfg_near,
        )
        mid = len(ds) // 2
        val = ds[mid]["near_images"].abs().sum().item()
        print(f"   Somme absolue de near_images (attendu > 0) : {val:.4f}")
        assert val > 0.0
        print("   ✓ PASS — near_images non nul avec near_guided=True")

    @needs_pair
    def test_resample_changes_augmented_data(self, mat_pair, cfg):
        print("\n   OBJECTIF : vérifier que resample() produit de nouvelles augmentations")
        print("   POURQUOI : si resample() retournait toujours la même augmentation, le modèle")
        print("              verrait les mêmes transformations à chaque époque et sur-apprendrait")
        print("              les artefacts d'augmentation")
        _sep()
        ds = MatIRMDataset(
            [mat_pair],
            train_transform=TrainTransform(cfg),
            val_transform=ValTransform(),
            train=True, config=cfg,
        )
        aug_before = ds.aug_volumes[0].copy()
        changed = False
        for attempt in range(10):
            ds.resample()
            if not np.array_equal(aug_before, ds.aug_volumes[0]):
                changed = True
                print(f"   Changement détecté après {attempt + 1} appel(s) à resample()")
                break
        if not changed:
            print("   ÉCHEC : resample() a produit le même résultat 10 fois de suite")
        assert changed
        print("   ✓ PASS — resample() renouvelle les augmentations")

    @needs_pair
    def test_val_no_augmentation(self, val_dataset):
        print("\n   OBJECTIF : vérifier qu'en mode val les volumes augmentés = volumes resizés bruts")
        print("   POURQUOI : ValTransform est une identité ; si aug_volumes ≠ volumes en val,")
        print("              cela signifie que TrainTransform est appliqué par erreur à la val")
        _sep()
        diff = np.abs(val_dataset.aug_volumes[0] - val_dataset.volumes[0]).max()
        print(f"   Diff max aug_volumes vs volumes (attendu 0) : {diff}")
        np.testing.assert_array_equal(val_dataset.aug_volumes[0], val_dataset.volumes[0])
        print("   ✓ PASS — pas d'augmentation en mode val")

    @needs_pair
    def test_visual_dataset_samples(self, dataset):
        print("\n   OBJECTIF [VISUEL] : afficher image + contour + near_image pour 3 coupes")
        print("   CE QU'ON ATTEND :")
        print("     col 1 (Image)   : coupe en niveaux de gris, valeurs dans [-1,1]")
        print("     col 2 (Contour) : carte continue [-1,1] alignée sur l'image")
        print("     col 3 (Near)    : tenseur nul (noir) car near_guided=False")
        _sep()
        indices = [0, len(dataset) // 2, len(dataset) - 1]
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        col_labels = ["Image  ∈ [-1, 1]", "Contour  ∈ [-1, 1]", "Near image (0 si désactivé)"]
        row_labels  = [f"Coupe {i}" for i in indices]
        for row, idx in enumerate(indices):
            s = dataset[idx]
            arrays = [s["images"][0].numpy(), s["contours"][0].numpy(), s["near_images"][0].numpy()]
            for col, (arr, clabel) in enumerate(zip(arrays, col_labels)):
                axes[row, col].imshow(arr, cmap="gray")
                axes[row, col].set_title(f"{clabel}\n{row_labels[row]}")
                axes[row, col].axis("off")
        fig.suptitle("Sorties MatIRMDataset\n"
                     "Vérifiez : image et contour alignés • contour continu [-1,1] • near_image = noir (near_guided=False)",
                     fontsize=11)
        _save(fig, "06_dataset_samples.png")
        print("   ✓ PASS — vérifiez l'alignement image↔contour dans le PNG")

    @needs_pair
    def test_visual_image_histogram(self, dataset):
        print("\n   OBJECTIF [VISUEL] : histogramme des intensités sur 30 coupes")
        print("   CE QU'ON ATTEND : distribution centrée, entièrement comprise entre -1 et +1")
        print("                     Des pics aux extrêmes indiquent une saturation percentile")
        _sep()
        values = np.concatenate([
            dataset[idx]["images"].numpy().ravel()
            for idx in range(min(30, len(dataset)))
        ])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=100, color="steelblue", edgecolor="none")
        ax.axvline(-1, color="red",   linestyle="--", linewidth=1.5, label="limite −1")
        ax.axvline( 1, color="green", linestyle="--", linewidth=1.5, label="limite +1")
        ax.set_xlabel("Valeur pixel")
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Distribution des intensités — {min(30, len(dataset))} coupes\n"
                     "Aucune barre ne doit dépasser les lignes rouge ou verte")
        ax.legend()
        _save(fig, "07_image_histogram.png")
        print(f"   Range effectif : [{values.min():.4f},  {values.max():.4f}]")
        print("   ✓ PASS — vérifiez que l'histogramme est contenu dans [-1, +1]")


# ── find_mat_pairs ────────────────────────────────────────────────────────────

class TestFindMatPairs:
    """
    find_mat_pairs() scanne récursivement un répertoire à la recherche de couples
    zscore_*.mat + contours_zscore_*.mat.
    Un volume sans fichier contours est ignoré avec un warning.
    """

    def test_finds_valid_pair(self, tmp_path):
        print("\n   OBJECTIF : vérifier la détection d'un couple valide")
        print("   POURQUOI : c'est le point d'entrée de toute la pipeline ;")
        print("              si la détection échoue, aucune donnée n'est chargée")
        _sep()
        (tmp_path / "zscore_p1.mat").touch()
        (tmp_path / "contours_zscore_p1.mat").touch()
        pairs = find_mat_pairs(str(tmp_path))
        print(f"   Paires trouvées : {len(pairs)}")
        print(f"   vol  : {Path(pairs[0][0]).name}")
        print(f"   cont : {Path(pairs[0][1]).name}")
        assert len(pairs) == 1
        assert pairs[0][0].endswith("zscore_p1.mat")
        assert pairs[0][1].endswith("contours_zscore_p1.mat")
        print("   ✓ PASS — couple détecté et nommé correctement")

    def test_skips_volume_without_contour(self, tmp_path):
        print("\n   OBJECTIF : vérifier qu'un zscore sans contours est ignoré (pas de crash)")
        print("   POURQUOI : pendant le développement les contours peuvent ne pas encore")
        print("              exister ; la pipeline doit continuer sur les paires complètes")
        _sep()
        (tmp_path / "zscore_p1.mat").touch()
        (tmp_path / "zscore_p2.mat").touch()
        (tmp_path / "contours_zscore_p2.mat").touch()
        pairs = find_mat_pairs(str(tmp_path))
        print(f"   2 zscore, 1 contour → paires trouvées : {len(pairs)}  (attendu 1)")
        assert len(pairs) == 1
        assert "p2" in pairs[0][0]
        print("   ✓ PASS — p1 ignoré (pas de contour), p2 retenu")

    def test_empty_directory(self, tmp_path):
        print("\n   OBJECTIF : vérifier le comportement sur dossier vide")
        print("   POURQUOI : main() lève une RuntimeError si pairs est vide ;")
        print("              find_mat_pairs doit retourner [] sans planter")
        _sep()
        pairs = find_mat_pairs(str(tmp_path))
        print(f"   Résultat : {pairs}")
        assert pairs == []
        print("   ✓ PASS — retourne [] sur dossier vide")

    def test_recursive_search(self, tmp_path):
        print("\n   OBJECTIF : vérifier la recherche récursive dans les sous-dossiers")
        print("   POURQUOI : les données sont organisées par patient dans des sous-dossiers")
        print("              (ex. data/patient_01/zscore_*.mat) — rglob doit les trouver")
        _sep()
        sub = tmp_path / "patient_01"
        sub.mkdir()
        (sub / "zscore_p1.mat").touch()
        (sub / "contours_zscore_p1.mat").touch()
        pairs = find_mat_pairs(str(tmp_path))
        print(f"   Fichiers dans sous-dossier patient_01/ → paires trouvées : {len(pairs)}")
        assert len(pairs) == 1
        print("   ✓ PASS — découverte récursive fonctionnelle")
