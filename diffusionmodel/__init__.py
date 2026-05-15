import logging
logger = logging.getLogger('base')

def create_SLF_model(opt):
    from .model import DDPM as M
    m = M(opt, data_type='SLF')
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_image_model(opt):
    from .model import DDPM as M
    m = M(opt, data_type='image')
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_PSD_model(opt):
    from .model import DDPM as M
    m = M(opt, data_type='PSD')
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_RM_model(opt):
    from .model import DDPM as M
    m = M(opt, data_type='RM')
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m